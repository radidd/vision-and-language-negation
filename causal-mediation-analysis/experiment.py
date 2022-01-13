import sys
sys.path.append('/UNITER')
import math
import statistics
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils_cma import batch, convert_results_to_pd

from model.nlvr2 import UniterForNlvr2Triplet

from utils.const import IMG_DIM

from apex import amp

np.random.seed(1)
torch.manual_seed(1)


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self,
                 ckpt_file,
                 model_config,
                 opts,
                 device='cuda',
                 output_attentions=False,
                 random_weights=False):
        super()
        self.device = device


        # Load UNITER finetuned for NLVR2 (triplet version)
        checkpoint = torch.load(ckpt_file)
        self.model = UniterForNlvr2Triplet(model_config, img_dim=IMG_DIM)
        self.model.init_type_embedding()
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model, _ = amp.initialize(self.model, enabled=opts.fp16, opt_level='O2')
        self.model.eval()

        # TODO: this does not work currently
        # if random_weights:
        #     print('Randomizing weights')
        #     self.model.init_weights()

        # Grab model configuration parameters
        self.num_layers = model_config.num_hidden_layers
        self.num_neurons = model_config.hidden_size

    def get_representations(self, context, position=0):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            representations[layer] = output[0][position]
        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            handles.append(self.model.uniter.embeddings.word_embeddings.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=-1)))
            # hidden layers; intervening on the output of the feedforward network
            for layer in range(self.num_layers):
                handles.append(self.model.uniter.encoder.layer[layer]\
                                   .output.dropout.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))

            logits = self.model(context, compute_loss=False)

            for h in handles:
                h.remove()

        return representation

    def get_probabilities_for_examples(self, context):
        """Return probabilities of single-token candidates given context"""

        logits = self.model(context, compute_loss = False)
        probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    def neuron_intervention(self,
                            context,
                            outputs,
                            rep,
                            layers,
                            neurons,
                            position,
                            intervention_type='replace',
                            alpha=1.):
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output,
                              position,
                              neurons,
                              intervention,
                              intervention_type):
            # Get the neurons to intervene on
            neurons = torch.LongTensor(neurons).to(self.device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            base = output[:, position, :].gather(
                1, neurons)
            intervention_view = intervention.view_as(base)

            if intervention_type == 'replace':
                base = intervention_view
            elif intervention_type == 'diff':
                base += intervention_view
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")
            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(output).byte()
            for i, v in enumerate(neurons):
                scatter_mask[i, position, v] = 1
            # Then take values from base and scatter
            output.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons)
        batch_context = {}
        for key, value in context.items():
            try:
                if len(value.shape) == 2:
                    batch_context[key] = value.repeat(batch_size,1)
                elif len(value.shape) == 3:
                    batch_context[key] = value.repeat(batch_size, 1, 1)
            except AttributeError:
                continue

        handle_list = []
        for layer in set(layers):
          neuron_loc = np.where(np.array(layers) == layer)[0]
          n_list = []
          for n in neurons:
            unsorted_n_list = [n[i] for i in neuron_loc]
            n_list.append(list(np.sort(unsorted_n_list)))
          intervention_rep = alpha * rep[layer][n_list]
          if layer == -1:
              wte_intervention_handle = self.model.uniter.embeddings.word_embeddings.register_forward_hook(
                  partial(intervention_hook,
                          position=position,
                          neurons=n_list,
                          intervention=intervention_rep,
                          intervention_type=intervention_type))
              handle_list.append(wte_intervention_handle)
          else:
              mlp_intervention_handle = self.model.uniter.encoder.layer[layer]\
                                            .output.dropout.register_forward_hook(
                  partial(intervention_hook,
                          position=position,
                          neurons=n_list,
                          intervention=intervention_rep,
                          intervention_type=intervention_type))
              handle_list.append(mlp_intervention_handle)
        new_probabilities = self.get_probabilities_for_examples(
            batch_context)
        for hndle in handle_list:
          hndle.remove()
        return new_probabilities


    def neuron_intervention_single_experiment(self,
                                              intervention,
                                              intervention_type, layers_to_adj=[],
                                              neurons_to_adj=[],
                                              alpha=100,
                                              bsize=500, intervention_loc='all'):
        """
        run one full neuron intervention experiment
        """

        with torch.no_grad():

            # Compute representations of the non-negated and negated versions; position 0 corresponds to the [CLS] token
            orig_representations = self.get_representations(
                intervention[0],
                position=0)
            negated_representations = self.get_representations(
                intervention[1],
                position=0)

            # e.g. There aren't two dogs.
            if intervention_type == 'negate_direct':
                context = intervention[1]
                rep = orig_representations
                replace_or_diff = 'replace'
            # e.g. There are two dogs.
            elif intervention_type == 'negate_indirect':
                context = intervention[0]
                rep = negated_representations
                replace_or_diff = 'replace'
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")

            # Probabilities without intervention (Base case)
            # Candidate 1 is False; Candidate 2 is True
            # TODO: this can be simplified since there are only two candidates
            candidate1_orig_prob, candidate2_orig_prob = self.get_probabilities_for_examples(
                intervention[0])[0]
            candidate1_neg_prob, candidate2_neg_prob = self.get_probabilities_for_examples(
                intervention[1])[0]

            # Running interventions
            if intervention_loc == 'all':
              candidate1_probs = torch.zeros((self.num_layers + 1, self.num_neurons))
              candidate2_probs = torch.zeros((self.num_layers + 1, self.num_neurons))

              for layer in range(-1, self.num_layers):
                for neurons in batch(range(self.num_neurons), bsize):
                    neurons_to_search = [[i] + neurons_to_adj for i in neurons]
                    layers_to_search = [layer] + layers_to_adj

                    probs = self.neuron_intervention(
                        context=context,
                        outputs=[0,1],
                        rep=rep,
                        layers=layers_to_search,
                        neurons=neurons_to_search,
                        position=0,
                        intervention_type=replace_or_diff,
                        alpha=alpha)

                    for neuron, (p1, p2) in zip(neurons, probs):
                        candidate1_probs[layer + 1][neuron] = p1
                        candidate2_probs[layer + 1][neuron] = p2

            elif intervention_loc == 'layer':
              layers_to_search = (len(neurons_to_adj) + 1)*[layers_to_adj]
              candidate1_probs = torch.zeros((1, self.num_neurons))
              candidate2_probs = torch.zeros((1, self.num_neurons))

              for neurons in batch(range(self.num_neurons), bsize):
                neurons_to_search = [[i] + neurons_to_adj for i in neurons]

                probs = self.neuron_intervention(
                    context=context,
                    outputs=[0,1],
                    rep=rep,
                    layers=layers_to_search,
                    neurons=neurons_to_search,
                    position=0,
                    intervention_type=replace_or_diff,
                    alpha=alpha)
                for neuron, (p1, p2) in zip(neurons, probs):
                    candidate1_probs[0][neuron] = p1
                    candidate2_probs[0][neuron] = p2
            else:
              probs = self.neuron_intervention(
                        context=context,
                        outputs=[0,1],
                        rep=rep,
                        layers=layers_to_adj,
                        neurons=neurons_to_adj,
                        position=0,
                        intervention_type=replace_or_diff,
                        alpha=alpha)
              for neuron, (p1, p2) in zip(neurons_to_adj, probs):
                  candidate1_probs = p1
                  candidate2_probs = p2


        return (candidate1_orig_prob, candidate2_orig_prob,
                candidate1_neg_prob, candidate2_neg_prob,
                candidate1_probs, candidate2_probs)
