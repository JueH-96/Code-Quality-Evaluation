def min_total_weirdness(N, K, lost_socks):
    # Create a set of lost sock colors for quick lookup
    lost_set = set(lost_socks)
    
    # Create a list of available socks
    available_socks = []
    for color in range(1, N + 1):
        if color not in lost_set:
            available_socks.append(color)
    
    # Calculate the number of pairs we can form
    total_socks = len(available_socks)
    pairs_count = total_socks // 2
    
    # If there are no pairs to form, weirdness is 0
    if pairs_count == 0:
        return 0
    
    # Calculate the minimum total weirdness
    weirdness = 0
    for i in range(pairs_count):
        weirdness += abs(available_socks[2 * i] - available_socks[2 * i + 1])
    
    return weirdness

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
lost_socks = list(map(int, data[2:2 + K]))

# Get the result and print it
result = min_total_weirdness(N, K, lost_socks)
print(result)