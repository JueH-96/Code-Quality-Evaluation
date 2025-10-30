import collections

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        
        # dp[i] will store the minimum number of valid strings
        # to form the suffix target[i:]
        # Initialize with infinity. dp[n] is 0 as an empty suffix requires 0 strings.
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        # Define TrieNode internally, as it's only used here.
        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
        
        trie_root = TrieNode()
        
        # Build the Trie from words.
        # Any path from the root to any node in the Trie represents a prefix
        # of at least one string in 'words', thus it's a "valid string".
        for word in words:
            node = trie_root
            for char in word:
                node = node.children[char] # defaultdict creates node if not exists
        
        # Fill dp table from right to left (bottom-up approach)
        # i iterates from n-1 down to 0, representing the start index of the current suffix
        for i in range(n - 1, -1, -1):
            current_trie_node = trie_root
            
            # Try to match prefixes of target[i:] using the Trie.
            # k iterates from i to n-1, representing the end index (exclusive) of the current prefix.
            # So target[i:k+1] is the current prefix being checked.
            for k in range(i, n):
                char = target[k]
                
                # Check if the current character 'char' exists as a child
                # of the current Trie node.
                if char not in current_trie_node.children:
                    # If not, no more valid prefixes can be formed starting with target[i]
                    # and continuing with target[k]. Break and move to the next 'i'.
                    break
                
                # Move to the next Trie node.
                current_trie_node = current_trie_node.children[char]
                
                # At this point, target[i:k+1] is a valid prefix (of some word in 'words').
                # We want to check if the remaining suffix target[k+1:] can be formed.
                # If dp[k+1] is reachable (not infinity), then we can potentially
                # form target[i:] using target[i:k+1] (1 string) and then target[k+1:].
                if dp[k+1] != float('inf'):
                    dp[i] = min(dp[i], 1 + dp[k+1])
        
        # The result is dp[0], which stores the minimum number of valid strings
        # to form the entire target string (target[0:]).
        # If dp[0] is still infinity, it means target cannot be formed.
        return dp[0] if dp[0] != float('inf') else -1