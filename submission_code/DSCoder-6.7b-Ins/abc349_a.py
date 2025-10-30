# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Calculate the total score of all players
total_score = sum(A)

# The final score of the last player is the total score minus the sum of the scores of all other players
final_score = total_score - sum(A[:N-1])

print(final_score)