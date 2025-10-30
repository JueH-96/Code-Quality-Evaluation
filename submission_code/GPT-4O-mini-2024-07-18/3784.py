from typing import List
from collections import Counter

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        answer = [0] * n
        
        for i in range(n):
            # Create a frequency counter excluding the i-th word
            counter = Counter(words[j] for j in range(n) if j != i)
            
            # If we have fewer than k distinct words, the answer is 0
            if len(counter) < k:
                continue
            
            # Get the k most common words
            most_common = counter.most_common(k)
            
            # Find the longest common prefix among the k most common words
            common_prefix_length = self.common_prefix_length([word for word, _ in most_common])
            answer[i] = common_prefix_length
        
        return answer
    
    def common_prefix_length(self, words: List[str]) -> int:
        if not words:
            return 0
        
        # Start with the first word as the prefix
        prefix = words[0]
        
        for word in words[1:]:
            # Update the prefix length based on the current word
            while not word.startswith(prefix):
                prefix = prefix[:-1]  # Reduce the prefix length
                if not prefix:
                    return 0
        
        return len(prefix)