class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If n is already equal to k, no changes are needed
        if n == k:
            return 0
        
        # If k is greater than n, it's impossible to make n equal to k by only changing 1s to 0s
        if k > n:
            return -1
        
        # Calculate the XOR of n and k to find the differing bits
        xor_result = n ^ k
        
        # Count the number of 1s in the XOR result, which represents the number of changes needed
        changes_needed = bin(xor_result).count('1')
        
        # Check if all changes needed are possible (i.e., all 1s in XOR result are in n)
        if xor_result & n == xor_result:
            return changes_needed
        else:
            return -1