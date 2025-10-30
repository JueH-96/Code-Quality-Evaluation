# YOUR CODE HERE
import sys

def min_weirdness(N, K, A):
    # Initialize the count of socks for each color
    socks = [2] * (N + 1)
    for a in A:
        socks[a] -= 1
    
    # Initialize the total weirdness
    total_weirdness = 0
    # Initialize the count of unpaired socks
    unpaired = 0
    
    # Iterate through the colors to form pairs
    for i in range(1, N + 1):
        if socks[i] == 2:
            # If there are two socks of the same color, form a pair
            total_weirdness += 0
            unpaired = 0
        elif socks[i] == 1:
            # If there is one sock of the current color
            if unpaired == 0:
                # If there is no unpaired sock from the previous color, keep it unpaired
                unpaired = 1
            else:
                # If there is an unpaired sock from the previous color, form a pair
                total_weirdness += abs(i - (i - 1))
                unpaired = 0
        else:
            # If there are no socks of the current color, do nothing
            pass
    
    # If there is an unpaired sock left, it will not be paired
    return total_weirdness

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Compute and print the result
print(min_weirdness(N, K, A))