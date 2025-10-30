from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        
        # Iterate through all possible pairs of indices (i, j) such that i < j
        # The outer loop iterates through the first string words[i]
        for i in range(n):
            # The inner loop iterates through the second string words[j].
            # We start j from i + 1 to ensure that j is always greater than i,
            # fulfilling the i < j condition.
            for j in range(i + 1, n):
                str1 = words[i]
                str2 = words[j]
                
                # Check if str1 is both a prefix and a suffix of str2.
                # According to the definition isPrefixAndSuffix(str1, str2) returns true if:
                # 1. The length of str1 is less than or equal to the length of str2.
                #    (A string cannot be a prefix or suffix of a strictly shorter string).
                # 2. str1 must match the beginning part of str2 (the prefix).
                # 3. str1 must match the ending part of str2 (the suffix).
                
                # Check condition 1: length constraint. If str1 is longer than str2, it cannot be a prefix or suffix.
                if len(str1) <= len(str2):
                    # Check condition 2: str1 is a prefix of str2
                    # Python's str.startswith() method is suitable for this.
                    is_prefix = str2.startswith(str1)
                    
                    # Check condition 3: str1 is a suffix of str2
                    # Python's str.endswith() method is suitable for this.
                    is_suffix = str2.endswith(str1)
                    
                    # If both the prefix and suffix conditions are true for the current pair (str1, str2),
                    # it means isPrefixAndSuffix(str1, str2) is true.
                    if is_prefix and is_suffix:
                        # Increment the count of valid pairs.
                        count += 1
                        
        # After checking all possible pairs (i, j) with i < j, return the total count.
        return count