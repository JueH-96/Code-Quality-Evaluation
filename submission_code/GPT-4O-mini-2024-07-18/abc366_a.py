# YOUR CODE HERE
def election_outcome(N, T, A):
    remaining_votes = N - (T + A)
    if T > A + remaining_votes:
        return "Yes"
    else:
        return "No"

import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
T = int(data[1])
A = int(data[2])

print(election_outcome(N, T, A))