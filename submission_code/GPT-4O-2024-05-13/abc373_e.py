# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

total_votes_so_far = sum(A)
remaining_votes = K - total_votes_so_far

results = []

for i in range(N):
    current_votes = A[i]
    other_votes = sorted(A[:i] + A[i+1:], reverse=True)
    
    if M == 1:
        results.append(0)
        continue
    
    if len(other_votes) < M - 1:
        results.append(0)
        continue
    
    min_votes_to_beat = other_votes[M-2]
    
    if current_votes > min_votes_to_beat:
        results.append(0)
    else:
        needed_votes = min_votes_to_beat - current_votes + 1
        if needed_votes <= remaining_votes:
            results.append(needed_votes)
        else:
            results.append(-1)

print(" ".join(map(str, results)))