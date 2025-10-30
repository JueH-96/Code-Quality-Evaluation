# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_full_moons(N, M, P):
    if M > N:
        return 0
    return (N - M) // P + 1

# Read input
data = input().strip().split()
N = int(data[0])
M = int(data[1])
P = int(data[2])

# Calculate and print the result
print(count_full_moons(N, M, P))