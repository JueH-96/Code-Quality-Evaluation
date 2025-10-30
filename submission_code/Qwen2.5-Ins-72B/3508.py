class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Calculate the XOR of n and k to find the differing bits
        xor_result = n ^ k
        
        # Count the number of 1s in the XOR result, which represents the differing bits
        diff_count = bin(xor_result).count('1')
        
        # If k has more 1s than n, it is impossible to make n equal to k by only changing 1s to 0s
        if bin(k).count('1') > bin(n).count('1'):
            return -1
        
        # The number of changes needed is the number of differing bits
        return diff_count