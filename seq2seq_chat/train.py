# encoding=utf-8
from torch import optim
from data_gen import ChatbotDataset
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, encoder_optimizer,
          decoder_optimizer):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def validate(val_loader, encoder, decoder):
    with torch.no_grad():
        # eval mode (no dropout or batchnorm)
        encoder.eval()
        decoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        print_losses = []
        n_totals = 0

        start = time.time()

        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        # Batches
        for i_batch in range(val_loader.__len__()):
            input_variable, lengths, target_variable, mask, max_target_len = val_loader.__getitem__(i_batch)
            # Set device options
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            target_variable = target_variable.to(device)
            mask = mask.to(device)

            # Initialize variables
            loss = 0

            # Forward pass through encoder
            encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:decoder.n_layers]

            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

            loss = sum(print_losses) / n_totals
            # Keep track of metrics
            losses.update(loss, max_target_len)
            batch_time.update(time.time() - start)

            start = time.time()

    return sum(print_losses) / n_totals


def evaluate(searcher, sentence, max_length=max_len):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens
                     if token != EOS_token and token != PAD_token]
    return decoded_words


def main():
    train_loader = ChatbotDataset('train')
    val_loader = ChatbotDataset('valid')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Initializations
    print('Initializing ...')
    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)
    print(train_loader.__len__())
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()

        start = time.time()

        # Batches
        for i in range(train_loader.__len__()):
            input_variable, lengths, target_variable, mask, max_target_len = train_loader.__getitem__(i)
            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                         encoder_optimizer, decoder_optimizer)

            # Keep track of metrics
            losses.update(loss, max_target_len)
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_every == 0:
                print('[{0}] Epoch: [{1}][{2}/{3}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(timestamp(), epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

        # One epoch's validation
        val_loss = validate(val_loader, encoder, decoder)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)
        for sentence in pick_n_valid_sentences(10):
            decoded_words = evaluate(searcher, sentence)
            print('Human: {}'.format(sentence))
            print('Bot: {}'.format(''.join(decoded_words)))

        # Save checkpoint
        if epoch % save_every == 0:
            directory = save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc.__dict__
            }, os.path.join(directory, '{}_{}_{}.tar'.format('checkpoint', epoch, val_loss)))


if __name__ == '__main__':
    main()
