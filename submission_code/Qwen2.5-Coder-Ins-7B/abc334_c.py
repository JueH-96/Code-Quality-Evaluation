# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Count the remaining socks of each color
remaining_socks = [0] * (N + 1)
for i in range(K):
    remaining_socks[A[i]] -= 1

# Calculate the total weirdness
total_weirdness = 0
for i in range(1, N + 1):
    remaining_socks[i] += remaining_socks[i - 1]
    if remaining_socks[i] > 0:
        total_weirdness += remaining_socks[i] - 1

# Since we are pairing, we need to divide by 2
total_weirdness //= 2

print(total_weirdness)