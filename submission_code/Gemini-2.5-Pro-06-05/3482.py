import collections

class Solution:
  def minimumCost(self, target: str, words: list[str], costs: list[int]) -> int:
    """
    Calculates the minimum cost to form the target string by concatenating words.

    This solution uses dynamic programming combined with the Aho-Corasick algorithm.
    1.  A DP array `dp[i]` stores the minimum cost to form the prefix `target[:i]`.
    2.  An Aho-Corasick automaton is built from the given `words` to efficiently 
        find all word matches ending at each position in the `target`.
    3.  The DP state `dp[i]` is updated by considering all word matches ending at `i`
        and using the costs from previously computed DP states.
    """
    
    # TrieNode for the Aho-Corasick automaton
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.fail = None
            # Stores (length, cost) of words ending exactly at this node
            self.word_info = []

    # 1. Preprocessing: Get unique words with their minimum costs.
    min_costs = {}
    for word, cost in zip(words, costs):
        if word not in min_costs or cost < min_costs[word]:
            min_costs[word] = cost

    # 2. Build the Trie part of the Aho-Corasick automaton.
    root = TrieNode()
    for word, cost in min_costs.items():
        node = root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.word_info.append((len(word), cost))

    # 3. Build failure links using a Breadth-First Search (BFS).
    queue = collections.deque()
    root.fail = root
    
    for node in root.children.values():
        node.fail = root
        queue.append(node)

    while queue:
        current_node = queue.popleft()
        for char, next_node in current_node.children.items():
            fail_node = current_node.fail
            while char not in fail_node.children and fail_node is not root:
                fail_node = fail_node.fail
            
            if char in fail_node.children:
                next_node.fail = fail_node.children[char]
            else:
                next_node.fail = root
            
            queue.append(next_node)

    # 4. Dynamic Programming using the Aho-Corasick automaton.
    n = len(target)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    current_node = root
    for i in range(1, n + 1):
        char = target[i - 1]
        
        # Follow transitions in the automaton. If a direct child doesn't exist,
        # follow failure links until a transition is found or we reach the root.
        while char not in current_node.children and current_node is not root:
            current_node = current_node.fail
        if char in current_node.children:
            current_node = current_node.children[char]
        
        # Check for all matches ending at position i by following the failure links
        # from the current node.
        match_node = current_node
        while match_node is not root:
            for length, cost in match_node.word_info:
                # If the prefix `target[:i-length]` is reachable
                if dp[i - length] != float('inf'):
                    dp[i] = min(dp[i], dp[i - length] + cost)
            match_node = match_node.fail

    # 5. Return the final result.
    result = dp[n]
    return int(result) if result != float('inf') else -1