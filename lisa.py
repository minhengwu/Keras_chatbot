from create_model import build_model
import numpy as np
import pickle

class Bot():
    def __init__(self, weight_path, index_path, word_path):
        num_encoder_tokens = 5004
        num_decoder_tokens = 5004
        latent_dim = 256
        _, self.encoder_model, self.decoder_model = build_model(latent_dim,num_encoder_tokens,num_decoder_tokens,weight_path=weight_path)
        with open(index_path, 'rb') as handle:
            self.word_to_idx = pickle.load(handle)
            handle.close()

        with open(word_path, 'rb') as handle:
            self.idx_to_word = pickle.load(handle)
            handle.close()


    def analyze(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 5004))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.word_to_idx['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.idx_to_word[sampled_token_index]
            decoded_sentence += (sampled_char + ' ')

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                        len(decoded_sentence) > 20):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 5004))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence