# YOUR CODE HERE
def min_total_weirdness(N, K, A):
    # Initialize the count of each color
    color_count = [2] * (N + 1)
    
    # Decrease the count for lost socks
    for a in A:
        color_count[a] -= 1
    
    # Collect all remaining socks
    remaining_socks = []
    for i in range(1, N + 1):
        if color_count[i] > 0:
            remaining_socks.extend([i] * color_count[i])
    
    # Sort the remaining socks
    remaining_socks.sort()
    
    # Calculate the minimum total weirdness
    total_weirdness = 0
    for i in range(0, len(remaining_socks) - 1, 2):
        total_weirdness += abs(remaining_socks[i] - remaining_socks[i + 1])
    
    return total_weirdness

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Calculate and print the result
print(min_total_weirdness(N, K, A))