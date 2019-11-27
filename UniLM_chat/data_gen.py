import codecs

# remove punctuations
def remove_punc(line):
    line = line.replace('。','')
    line = line.replace('？','')
    line = line.replace('！','')
    line = line.replace('，','')
    line = line.replace('.','')
    line = line.replace(',','')
    line = line.replace('?','')
    line = line.replace('!','')
    line = line.replace('“','')
    line = line.replace('”','')
    line = line.replace('¥', '')
    line = line.replace('@', '')
    line = line.replace('\n', '')
    line = line.replace('(', '')
    line = line.replace(')', '')
    return line


def generate_single_pairs_from_multi_turn(utterances):
    pairs = []
    for index in range(len(utterances) - 1):
        pairs.append((utterances[index], utterances[index + 1]))
    return pairs

def generate_dataset():

    start_end_symbol = "E"
    utterance_symbol = "M"

    corpus = []

    raw_corpus_file = codecs.open("./data/xiaohuangji50w_nofenci.conv", encoding="utf-8", errors="replace")

    single_session = []
    session_lengths = []

    for index, line in enumerate(raw_corpus_file):
        if index % 100000 == 0:
            print("xiaohuangji50w_nofenci", index)

        if line.startswith(start_end_symbol):
            if len(single_session) == 2:
                pairs = generate_single_pairs_from_multi_turn(single_session)
                for pair in pairs:
                    if '小通' not in pair[0] and '小通' not in pair[1]: 
                        corpus.append([pair[0],pair[1]])
                session_lengths.append(len(single_session))
            single_session = []
        elif line.startswith(utterance_symbol):
            line = line[1:].strip()
            utterance = line.strip()
            single_session.append(utterance)
    raw_corpus_file.close()

    raw_corpus_file = codecs.open("./data/qinyun.csv", encoding="utf-8", errors="replace")
    for index, line in enumerate(raw_corpus_file):
        if index % 100000 == 0:
            print("qingyun", index)
        if "{" in line or "qq" in line or "菲菲" in line:
            continue
        pair = line.strip().split("|")
        corpus.append([pair[0][:-1], pair[1][1:]])
    raw_corpus_file.close()

    return corpus


