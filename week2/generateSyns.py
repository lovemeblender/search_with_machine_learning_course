import fasttext

model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
synonyms_file = open("/workspace/datasets/fasttext/synonyms.csv", "w")

neighbor_arr = []
for word in open('/workspace/datasets/fasttext/top_words.txt', 'r'):
    word_neighbors = model.get_nearest_neighbors(word)
    # Getting syns for word
    neighbor_arr = [word_syn for similarity, word_syn in word_neighbors if similarity > 0.8]
    if len(neighbor_arr) > 0: 
        synonyms_file.write(word + "," + ",".join(neighbor_arr) + '\n')
