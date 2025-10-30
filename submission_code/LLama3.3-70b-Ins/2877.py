from itertools import permutations

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # Generate all permutations of the input strings
        perms = list(permutations([a, b, c]))
        
        # Initialize the minimum length and the result string
        min_length = float('inf')
        result = ""
        
        # Iterate over all permutations
        for perm in perms:
            # Initialize the current string
            curr_str = ""
            
            # Iterate over the strings in the current permutation
            for i in range(len(perm)):
                # If the current string is empty, add the current string
                if not curr_str:
                    curr_str += perm[i]
                else:
                    # Find the maximum overlap between the current string and the next string
                    max_overlap = 0
                    for j in range(min(len(curr_str), len(perm[i])), 0, -1):
                        if curr_str[-j:] == perm[i][:j]:
                            max_overlap = j
                            break
                    
                    # Add the non-overlapping part of the next string to the current string
                    curr_str += perm[i][max_overlap:]
            
            # Update the minimum length and the result string if necessary
            if len(curr_str) < min_length:
                min_length = len(curr_str)
                result = curr_str
            elif len(curr_str) == min_length and curr_str < result:
                result = curr_str
        
        return result