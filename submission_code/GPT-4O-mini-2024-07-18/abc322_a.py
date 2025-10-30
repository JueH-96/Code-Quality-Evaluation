# YOUR CODE HERE
def find_abc_position(N, S):
    for i in range(N - 2):
        if S[i:i + 3] == "ABC":
            return i + 1  # Convert to 1-based index
    return -1

import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
S = data[1]

result = find_abc_position(N, S)
print(result)