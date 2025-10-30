class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Initialize the permutation difference to 0
        permutation_difference = 0
        
        # Iterate over each character in s
        for char in s:
            # Find the index of the character in s and t
            index_in_s = s.index(char)
            index_in_t = t.index(char)
            
            # Calculate the absolute difference and add it to the permutation difference
            permutation_difference += abs(index_in_s - index_in_t)
        
        return permutation_difference