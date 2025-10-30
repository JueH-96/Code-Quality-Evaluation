from collections import deque
from typing import List

class TrieNode:
    __slots__ = ['transitions', 'failure', 'output']
    
    def __init__(self):
        self.transitions = {}
        self.failure = None
        self.output = []  # list of (length, cost)

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Preprocess words to keep the minimal cost for each unique word
        unique_words = {}
        for word, cost in zip(words, costs):
            if word in unique_words:
                if cost < unique_words[word]:
                    unique_words[word] = cost
            else:
                unique_words[word] = cost
        
        # Build the trie structure
        root = TrieNode()
        for word, cost in unique_words.items():
            current = root
            for c in word:
                if c not in current.transitions:
                    current.transitions[c] = TrieNode()
                current = current.transitions[c]
            # Update end_info only if it's better (though unique_words ensures it's unique)
            if not hasattr(current, 'end_info'):
                current.end_info = (len(word), cost)
            else:
                # This case should not occur since unique_words are unique
                existing_cost = current.end_info[1]
                if cost < existing_cost:
                    current.end_info = (len(word), cost)
        
        # Build failure links and output lists using BFS
        queue = deque()
        root.failure = None
        
        # Initialize the children of the root
        for child in root.transitions.values():
            child.failure = root
            child.output = [child.end_info] if hasattr(child, 'end_info') else []
            if child.failure is not None:
                child.output += child.failure.output
            queue.append(child)
        
        # Process other nodes level by level
        while queue:
            current_node = queue.popleft()
            
            for char, child in current_node.transitions.items():
                # Find the failure node for this child
                failure_node = current_node.failure
                while failure_node is not None and char not in failure_node.transitions:
                    failure_node = failure_node.failure
                if failure_node is None:
                    child_failure = root
                else:
                    child_failure = failure_node.transitions.get(char, root)
                child.failure = child_failure
                
                # Prepare the output list for this child
                child.output = []
                if hasattr(child, 'end_info'):
                    child.output.append(child.end_info)
                if child.failure is not None:
                    child.output += child.failure.output
                
                queue.append(child)
        
        # Initialize DP array
        n = len(target)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        current_node = root
        
        for i in range(n):
            current_char = target[i]
            # Traverse the trie, following failure links as needed
            while current_node is not None and current_char not in current_node.transitions:
                current_node = current_node.failure
            if current_node is None:
                current_node = root
            else:
                current_node = current_node.transitions[current_char]
            
            # Update DP based on all possible words ending at this position
            for length, cost in current_node.output:
                start = i - length + 1
                if start >= 0 and dp[start] + cost < dp[i + 1]:
                    dp[i + 1] = dp[start] + cost
        
        return dp[n] if dp[n] != INF else -1