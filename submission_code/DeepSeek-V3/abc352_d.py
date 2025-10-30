# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    
    # Map each value to its index
    pos = {v: i for i, v in enumerate(P)}
    
    # We need to find a window of size K where the values are consecutive
    # So, for each possible starting value a, we check if a, a+1, ..., a+K-1 are in the permutation
    # and then find the minimal difference between the last and first index in the window
    
    # Since P is a permutation, all values from 1 to N are present
    # So, for each a, we can check if a, a+1, ..., a+K-1 are in P
    
    # To find the minimal i_K - i_1, we need to find the smallest window that contains a consecutive sequence of K elements
    
    # We can iterate over all possible a, and for each a, find the positions of a, a+1, ..., a+K-1
    # Then, the window size is the difference between the maximum and minimum index in these positions
    
    # Since N can be up to 2e5, we need an efficient way to find the positions
    
    # Precompute the positions of all values
    # pos is already a dictionary mapping value to index
    
    min_diff = float('inf')
    
    # Iterate over all possible starting values a
    # a can be from 1 to N - K + 1
    for a in range(1, N - K + 2):
        # Get the positions of a, a+1, ..., a+K-1
        indices = []
        valid = True
        for i in range(K):
            val = a + i
            if val not in pos:
                valid = False
                break
            indices.append(pos[val])
        if not valid:
            continue
        # Sort the indices to find the window
        indices.sort()
        current_diff = indices[-1] - indices[0]
        if current_diff < min_diff:
            min_diff = current_diff
        if min_diff == K - 1:
            break  # cannot be smaller
    
    print(min_diff)

if __name__ == "__main__":
    main()