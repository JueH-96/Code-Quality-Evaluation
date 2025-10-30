import sys

INF = 2000005

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.min_len = INF

def add_word(root, word):
    node = root
    path = [root]
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
        path.append(node)
    path[-1].is_end_of_word = True
    L = len(word)
    for p_node in path:
        p_node.min_len = min(p_node.min_len, L)

def compute_min_cost(root, S_k):
    len_S = len(S_k)
    min_cost_val = 10**9
    current_node = root
    d = 0
    while True:
        if d < len_S and S_k[d] in current_node.children:
            val_min_len = INF
            if current_node.is_end_of_word:
                val_min_len = min(val_min_len, d)
            for label, child in current_node.children.items():
                if label != S_k[d]:
                    val_min_len = min(val_min_len, child.min_len)
        else:
            val_min_len = current_node.min_len
        if val_min_len < INF:
            cost_p = len_S + val_min_len - 2 * d
            if cost_p < min_cost_val:
                min_cost_val = min(min_cost_val, cost_p)
        if d < len_S and S_k[d] in current_node.children:
            current_node = current_node.children[S_k[d]]
            d += 1
        else:
            break
    return min_cost_val

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S_list = []
for _ in range(N):
    S = data[index]
    index += 1
    S_list.append(S)

# Create trie root and set empty string option
root = TrieNode()
root.is_end_of_word = True
root.min_len = 0

# For each string, compute min cost and add to trie
for S_k in S_list:
    cost = compute_min_cost(root, S_k)
    print(cost)
    add_word(root, S_k)