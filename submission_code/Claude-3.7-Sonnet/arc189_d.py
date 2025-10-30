from collections import deque

def max_size(N, sizes, K):
    # K is 1-indexed, so we need to convert to 0-indexed
    # Takahashi's initial size
    takahashi_size = sizes[K-1]
    
    # Create deques for the slimes to the left and right of Takahashi
    left_slimes = deque(sizes[:K-1])  # Slimes to the left of Takahashi
    right_slimes = deque(sizes[K:])  # Slimes to the right of Takahashi
    
    # Continue until no more absorptions are possible
    while True:
        # Check if Takahashi can absorb the slimes
        can_absorb_left = len(left_slimes) > 0 and left_slimes[-1] < takahashi_size
        can_absorb_right = len(right_slimes) > 0 and right_slimes[0] < takahashi_size
        
        if not can_absorb_left and not can_absorb_right:
            break
        
        # Choose the larger slime to absorb if both are absorb-able
        if can_absorb_left and can_absorb_right:
            if left_slimes[-1] >= right_slimes[0]:
                takahashi_size += left_slimes[-1]
                left_slimes.pop()
            else:
                takahashi_size += right_slimes[0]
                right_slimes.popleft()
        elif can_absorb_left:
            takahashi_size += left_slimes[-1]
            left_slimes.pop()
        else:  # can_absorb_right
            takahashi_size += right_slimes[0]
            right_slimes.popleft()
    
    return takahashi_size

# Read the input
N = int(input())
sizes = list(map(int, input().split()))

# Calculate maximum size for each starting position K
results = []
for K in range(1, N+1):
    results.append(max_size(N, sizes, K))

# Print the results
print(" ".join(map(str, results)))