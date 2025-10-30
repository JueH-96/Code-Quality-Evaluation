# YOUR CODE HERE
def find_abc_position(N, S):
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            return i + 1
    return -1

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Find and print the position
print(find_abc_position(N, S))