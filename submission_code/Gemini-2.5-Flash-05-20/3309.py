from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)

        # Iterate through all possible indices i
        for i in range(n):
            # Iterate through all possible indices j such that j > i
            for j in range(i + 1, n):
                str1 = words[i]
                str2 = words[j]
                
                # Check if str1 is a prefix of str2
                is_prefix = str2.startswith(str1)
                
                # Check if str1 is a suffix of str2
                is_suffix = str2.endswith(str1)
                
                # If both conditions are true, increment the counter
                if is_prefix and is_suffix:
                    count += 1
        
        return count