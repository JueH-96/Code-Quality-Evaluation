from typing import List
import collections

class TrieNode:
    def __init__(self):
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            
class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a trie that represents all prefixes of any word.
        trie = Trie()
        for word in words:
            # When inserting the word into the trie, every prefix
            # becomes represented in the trie.
            trie.insert(word)
        
        n = len(target)
        # dp[i] = minimum count of valid strings to form target[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Try to form every substring starting at index i using valid prefix strings.
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            node = trie.root
            j = i
            # Extend the substring as long as it matches some prefix in our trie.
            while j < n and target[j] in node.children:
                node = node.children[target[j]]
                # This means target[i:j+1] is a valid prefix from one of the words.
                if dp[j+1] > dp[i] + 1:
                    dp[j+1] = dp[i] + 1
                j += 1
                
        return dp[n] if dp[n] != float('inf') else -1

# Code for local testing (Optional)
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    words1 = ["abc", "aaaaa", "bcdef"]
    target1 = "aabcdabc"
    print(sol.minValidStrings(words1, target1))  # Output: 3

    # Example 2
    words2 = ["abababab", "ab"]
    target2 = "ababaababa"
    print(sol.minValidStrings(words2, target2))  # Output: 2

    # Example 3
    words3 = ["abcdef"]
    target3 = "xyz"
    print(sol.minValidStrings(words3, target3))  # Output: -1