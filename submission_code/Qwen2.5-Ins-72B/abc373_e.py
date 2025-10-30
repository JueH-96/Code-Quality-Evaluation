# YOUR CODE HERE
import sys

def solve_election(N, M, K, A):
    remaining_votes = K - sum(A)
    A_sorted = sorted(A, reverse=True)
    result = []
    
    for i in range(N):
        votes_needed = 0
        current_votes = A[i]
        rank = sum(1 for x in A if x > current_votes)
        
        if rank < M:
            result.append(0)
            continue
        
        if rank - M >= N - 1:
            result.append(-1)
            continue
        
        for j in range(M - 1):
            if A_sorted[j] > current_votes:
                votes_needed += A_sorted[j] - current_votes + 1
                current_votes = A_sorted[j] + 1
        
        if votes_needed > remaining_votes:
            result.append(-1)
        else:
            result.append(votes_needed)
    
    return result

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

result = solve_election(N, M, K, A)
print(' '.join(map(str, result)))