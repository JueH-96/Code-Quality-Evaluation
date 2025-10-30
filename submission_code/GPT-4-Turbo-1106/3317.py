from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character in all words combined
        char_freq = Counter()
        for word in words:
            char_freq.update(word)
        
        # Count the number of odd frequencies
        odd_count = sum(1 for freq in char_freq.values() if freq % 2 != 0)
        
        # The maximum number of palindromes is the number of words minus the number of odd frequencies
        # divided by 2, because we can pair up two odd frequencies to make them even.
        # If there are no odd frequencies, all words can be palindromes.
        return len(words) - max(0, (odd_count - 1) // 2)

# Example usage:
# sol = Solution()
# print(sol.maxPalindromesAfterOperations(["abbb","ba","aa"])) # Output: 3
# print(sol.maxPalindromesAfterOperations(["abc","ab"])) # Output: 2
# print(sol.maxPalindromesAfterOperations(["cd","ef","a"])) # Output: 1