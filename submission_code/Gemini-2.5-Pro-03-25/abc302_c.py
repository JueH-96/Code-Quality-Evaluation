# YOUR CODE HERE
import sys
from itertools import permutations

# Function to calculate Hamming distance between two strings
# s1 and s2 are strings assumed to be of equal length M
def hamming_distance(s1, s2, M):
    """Calculates the Hamming distance between two strings s1 and s2 of length M.
    The Hamming distance is the number of positions at which the corresponding characters are different.
    """
    dist = 0
    # Iterate through each character position from 0 to M-1
    for k in range(M):
        # If characters at the current position differ, increment the distance counter
        if s1[k] != s2[k]:
            dist += 1
    # Return the total count of differing positions
    return dist

def solve():
    """Reads input strings, checks all possible orderings (permutations) 
    to see if any satisfy the condition, and prints the result ("Yes" or "No").
    The condition is that for a sequence T_1, ..., T_N, every adjacent pair (T_i, T_{i+1}) 
    must differ by exactly one character (Hamming distance of 1).
    """
    # Read N (number of strings) and M (length of each string) from standard input
    # Input format is "N M" on the first line
    N, M = map(int, sys.stdin.readline().split())
    
    # Read N strings from standard input into a list S
    # Each string is on a new line
    # `strip()` removes leading/trailing whitespace, including the newline character
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Generate a list of indices [0, 1, ..., N-1] corresponding to the strings in list S
    indices = list(range(N))
    
    # Flag to keep track if a valid permutation (ordering) has been found
    found_valid_permutation = False
    
    # Iterate through all possible permutations of the indices
    # `itertools.permutations(indices)` generates all N! possible orderings of the indices
    # This corresponds to checking all possible arrangements of the given N strings
    for p in permutations(indices):
        # Each `p` is a tuple representing an ordering of the strings, e.g., (p[0], p[1], ..., p[N-1])
        # This tuple contains indices referring to the strings in the list S.
        
        # Assume the current permutation `p` represents a valid sequence of strings according to the condition
        is_valid_sequence = True
        
        # Check the condition for all adjacent pairs in the sequence defined by permutation `p`
        # We need to check N-1 pairs: (T_0, T_1), (T_1, T_2), ..., (T_{N-2}, T_{N-1}),
        # where T_i is the string S[p[i]] corresponding to the i-th element in the permutation.
        for i in range(N - 1):
            # Get the indices from the permutation `p` for the two adjacent strings
            idx1 = p[i]   # Index of the i-th string in the current sequence
            idx2 = p[i+1] # Index of the (i+1)-th string in the current sequence
            
            # Retrieve the actual strings from list `S` using these indices
            string1 = S[idx1]
            string2 = S[idx2]
            
            # Calculate the Hamming distance between the two adjacent strings
            # If the distance is not exactly 1, this sequence arrangement is invalid according to the problem condition
            if hamming_distance(string1, string2, M) != 1:
                # Mark this permutation as invalid
                is_valid_sequence = False
                # If an invalid pair is found, there's no need to check the remaining pairs for this permutation
                # Move on to the next permutation
                break 
        
        # After checking all adjacent pairs for the current permutation `p`:
        # If `is_valid_sequence` is still True, it means all pairs satisfied the Hamming distance 1 condition.
        if is_valid_sequence:
            # A valid permutation fulfilling the condition has been found
            found_valid_permutation = True
            # Since the problem only asks if *at least one* such arrangement exists,
            # we can stop searching as soon as we find one.
            break

    # After the loop finishes (either by checking all permutations or breaking early):
    # Print the final result based on whether a valid permutation was found
    if found_valid_permutation:
        # If a valid permutation was found, print "Yes"
        print("Yes")
    else:
        # If no valid permutation was found after checking all N! possibilities, print "No"
        print("No")

# This is the standard entry point for Python scripts.
# It ensures that the `solve()` function is called when the script is executed directly.
if __name__ == '__main__':
    solve()