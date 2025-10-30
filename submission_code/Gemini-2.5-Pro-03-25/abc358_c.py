# # YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N (number of stands) and M (number of flavors) from input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the N strings representing flavors available at each stand
    # S[i] is the string for stand i (0-indexed based on list position)
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Precompute the bitmask for each stand. 
    # The j-th bit (0-indexed) of the mask is 1 if the stand sells flavor j+1, and 0 otherwise.
    stand_masks = []
    for i in range(N):
        mask = 0
        # Iterate through flavors j from 0 to M-1
        for j in range(M):
            # If stand i sells flavor j+1 (character at index j is 'o')
            if S[i][j] == 'o':
                # Set the j-th bit in the mask for stand i
                mask |= (1 << j)
        stand_masks.append(mask)

    # Initialize the minimum number of stands needed to a value larger than possible (N+1).
    # Since we are guaranteed a solution exists (worst case N stands), this initial value will be updated.
    min_stands = N + 1 
    
    # Calculate the target mask representing all M flavors.
    # This mask has the first M bits set to 1. It's equal to 2^M - 1.
    target_mask = (1 << M) - 1

    # Iterate through all possible non-empty subsets of stands.
    # A subset can be represented by an integer `i` from 1 to 2^N - 1.
    # If the j-th bit of `i` is 1, it means stand j (0-indexed) is included in the subset.
    for i in range(1, 1 << N):
        # Initialize the combined mask for the current subset of stands to 0.
        current_mask = 0
        
        # Iterate through each stand index j from 0 to N-1
        for j in range(N):
            # Check if stand j is included in the current subset `i`
            # This is done by checking if the j-th bit of `i` is set using bitwise AND.
            if (i >> j) & 1:
                # If stand j is included, combine its flavor mask with the current subset's mask
                # using bitwise OR. This accumulates all flavors available from the selected stands.
                current_mask |= stand_masks[j]
        
        # After combining masks for all stands in the subset, check if the result covers all flavors.
        # This means the combined mask should be equal to the target mask.
        if current_mask == target_mask:
            # If all flavors are covered, calculate the number of stands in this subset.
            # This is equal to the number of set bits (1s) in the integer `i`.
            # The bin() function converts an integer to its binary string representation (e.g., bin(5) -> '0b101').
            # .count('1') counts the occurrences of '1' in that string.
            num_stands = bin(i).count('1')
            
            # Update the minimum number of stands required if the current subset uses fewer stands.
            min_stands = min(min_stands, num_stands)

    # After checking all possible non-empty subsets, print the overall minimum number of stands found.
    print(min_stands)

# Call the solve function to execute the logic
solve()