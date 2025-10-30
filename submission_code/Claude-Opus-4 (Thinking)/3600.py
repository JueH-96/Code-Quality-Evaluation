class Solution:
    def kthCharacter(self, k: int) -> str:
        # Count the number of 1's in the binary representation of k-1
        increments = bin(k - 1).count('1')
        
        # Apply the increments to 'a' with wrap-around
        return chr(ord('a') + increments % 26)