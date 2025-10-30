class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def find_max_lcp_and_check_prefix(self, word):
        node = self.root
        lcp_length = 0
        is_prefix_exists = False
        
        for i, char in enumerate(word):
            if char not in node.children:
                break
            node = node.children[char]
            lcp_length = i + 1
            if node.is_end:
                is_prefix_exists = True
        
        return lcp_length, is_prefix_exists

n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

trie = Trie()

for k in range(n):
    current = strings[k]
    min_cost = len(current)  # Cost to make it empty
    
    if k > 0:
        # Find the maximum LCP with any previous string and check if any previous string is a prefix
        max_lcp, is_prefix = trie.find_max_lcp_and_check_prefix(current)
        
        if is_prefix:
            # If some previous string is a prefix of current, we only need to delete the suffix
            min_cost = min(min_cost, len(current) - max_lcp)
        else:
            # Otherwise, we need to consider transforming to strings with common prefix
            # The minimum cost would be achieved with the string that shares the longest common prefix
            if max_lcp > 0:
                # Cost = delete (len(current) - max_lcp) + add at least 1 character
                # But we need to find the actual minimum among all previous strings
                for j in range(k):
                    prev = strings[j]
                    # Calculate LCP between current and prev
                    lcp = 0
                    for i in range(min(len(current), len(prev))):
                        if current[i] == prev[i]:
                            lcp += 1
                        else:
                            break
                    
                    if prev == current[:len(prev)]:
                        # prev is a prefix of current
                        cost = len(current) - len(prev)
                    else:
                        # General case
                        cost = len(current) + len(prev) - 2 * lcp
                    
                    min_cost = min(min_cost, cost)
    
    print(min_cost)
    trie.insert(current)