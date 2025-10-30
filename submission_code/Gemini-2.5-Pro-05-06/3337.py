class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count the number of occurrences of character 'c' in string 's'.
        # Let this count be k.
        count_c = s.count(c)
        
        # If character 'c' appears k times in the string, any pair of these occurrences
        # (say, the i-th occurrence and the j-th occurrence, where i <= j)
        # defines a substring that starts with 'c' and ends with 'c'.
        #
        # The number of such pairs is the sum of integers from 1 to k:
        # k (pairs starting with the 1st 'c') +
        # k-1 (pairs starting with the 2nd 'c') +
        # ... +
        # 1 (pair starting with the k-th 'c').
        # This sum is equal to k * (k + 1) / 2, which is the k-th triangular number.
        #
        # For example:
        # If k = 3 (c appears 3 times), sum = 1 + 2 + 3 = 6. Formula: 3 * (3 + 1) // 2 = 6.
        # If k = 1 (c appears 1 time), sum = 1. Formula: 1 * (1 + 1) // 2 = 1.
        # If k = 0 (c does not appear), sum = 0. Formula: 0 * (0 + 1) // 2 = 0.
        # The formula holds for all non-negative k.
        
        # We use integer division // because the result must be an integer.
        # k * (k + 1) is always even, so the division by 2 is exact.
        result = count_c * (count_c + 1) // 2
        
        return result