
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *
from torch import optim

model_save_pth = "/data/nlp/chat/Chatbot-master/models/checkpoint_200_12.712512557527548.tar"

def main():
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    model = torch.load(model_save_pth, 'cpu')

    encoder.load_state_dict(torch.load(model_save_pth, device)['en'])
    decoder.load_state_dict(torch.load(model_save_pth, device)['de'])

    #encoder = model['en']
    #decoder = model.LuongAttnDecoderRNN['de']

    #encoder = encoder.to(device)
    #decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)

    for sentence in pick_n_valid_sentences(10):
        decoded_words = evaluate(searcher, sentence)
        print('Human: {}'.format(sentence))
        print('Bot: {}'.format(''.join(decoded_words)))


if __name__ == '__main__':
    main()

