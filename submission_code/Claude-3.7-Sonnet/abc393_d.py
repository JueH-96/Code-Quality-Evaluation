def min_swaps_to_contiguous(S):
    N = len(S)
    # Find positions of all 1s
    positions_of_ones = [i for i, c in enumerate(S) if c == '1']
    k = len(positions_of_ones)  # Number of 1s
    
    min_swaps = float('inf')
    
    # Try all possible contiguous positions for the 1s
    for l in range(N - k + 1):
        # Calculate swaps needed for each starting position l
        swaps = 0
        for i in range(k):
            # Calculate displacement from current position to target position
            swaps += abs(positions_of_ones[i] - (l + i))
        
        min_swaps = min(min_swaps, swaps)
    
    return min_swaps

# Read input
N = int(input())
S = input().strip()

# Calculate and print the answer
print(min_swaps_to_contiguous(S))