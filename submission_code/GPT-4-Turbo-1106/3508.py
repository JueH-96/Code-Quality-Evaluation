class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k is greater than n, it's impossible to make n equal to k by changing 1s to 0s
        if k > n:
            return -1
        
        # Count the number of bits that are different between n and k
        different_bits = n ^ k
        
        # Count the number of 1s in n that are not in k
        changes_needed = bin(n & different_bits).count('1')
        
        return changes_needed