import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# The final score of person N can be determined by the sum of the scores of all other players
# since the total sum of all scores must be zero.
total_score = sum(A)
final_score_N = -total_score

print(final_score_N)