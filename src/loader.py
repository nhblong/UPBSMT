import numpy as np
import torch
import io

from .dictionary import Dictionary


def load_fasttext_model(path):
    """
    Load a binarized fastText model.
    """
    try:
        import fasttext
    except ImportError:
        raise Exception("Unable to import fastText. Please install fastText for Python: "
                        "https://github.com/facebookresearch/fastText")
    return fasttext.load_model(path)

def load_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if params.max_vocab > 0 and len(word2id) >= params.max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings

def load_pth_embeddings(params, source):
    """
    Reload pretrained embeddings from a PyTorch binary file.
    """
    # reload PyTorch binary file
    lang = params.src_lang if source else params.tgt_lang
    data = torch.load(params.src_emb if source else params.tgt_emb)
    dico = data['dico']
    embeddings = data['vectors']
    assert dico.lang == lang
    assert embeddings.size(0) == len(dico)
    print("Loaded %i pre-trained word embeddings." % len(dico))
    return dico, embeddings


def load_bin_embeddings(params, source):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = params.src_lang if source else params.tgt_lang
    model = load_fasttext_model(params.src_emb if source else params.tgt_emb)
    words = model.get_labels()
    print("Loaded binary model. Generating embeddings ...")
    embeddings = torch.from_numpy(np.concatenate([model.get_word_vector(w)[None] for w in words], 0))
    print("Generated embeddings for %i words." % len(words))

    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    return dico, embeddings


def load_embeddings(params, source):
    """
    Reload aligned pretrained embeddings.
    """
    assert type(source) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    if emb_path.endswith('.pth'):
        dico, emb = load_pth_embeddings(params, source)
    if emb_path.endswith('.bin'):
        dico, emb = load_bin_embeddings(params, source)
    if emb_path.endswith('.vec'):
        dico, emd = load_txt_embeddings(params, source, False)
    if params.max_vocab > 0:
        dico.prune(params.max_vocab)
        emb = emb[:params.max_vocab]
    emb = emb.cuda()
    emb = emb / emb.norm(2, 1, keepdim=True).expand_as(emb)
    return dico, emb
