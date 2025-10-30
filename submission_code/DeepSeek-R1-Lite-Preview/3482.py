class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.fail = None
        self.output = []  # list of word indices that end at this node

def build_automaton(words):
    root = TrieNode()
    # Insert all words into the trie
    for idx, word in enumerate(words):
        node = root
        for char in word:
            c = ord(char) - ord('a')
            if node.children[c] is None:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.output.append(idx)
    # Build failure links
    queue = []
    for child in root.children:
        if child:
            child.fail = root
            queue.append(child)
    while queue:
        current = queue.pop(0)
        for c in range(26):
            child = current.children[c]
            if child:
                fail_node = current.fail
                while fail_node and fail_node.children[c] is None:
                    fail_node = fail_node.fail
                child.fail = fail_node.children[c] if fail_node and fail_node.children[c] else root
                child.output += child.fail.output
                queue.append(child)
    return root

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        root = build_automaton(words)
        N = len(target)
        dp = [float('inf')] * (N + 1)
        dp[0] = 0
        current = root
        for i in range(N):
            char = target[i]
            c = ord(char) - ord('a')
            # Transition to the correct state
            while current and current.children[c] is None:
                current = current.fail
            if current.children[c]:
                current = current.children[c]
                # Update dp for each word that ends here
                for word_idx in current.output:
                    L = len(words[word_idx])
                    j = i - L + 1
                    if j >= 0:
                        dp[i + 1] = min(dp[i + 1], dp[j] + costs[word_idx])
            else:
                current = root
        return dp[N] if dp[N] != float('inf') else -1