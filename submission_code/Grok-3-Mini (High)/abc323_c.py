import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
A = [int(data[index + i]) for i in range(M)]
index += M
S = [data[index + i] for i in range(N)]

# Compute current totals for all players
current_totals = [0] * N
for i in range(N):
    sum_solved = 0
    for j in range(M):
        if S[i][j] == 'o':
            sum_solved += A[j]
    current_totals[i] = sum_solved + (i + 1)  # bonus is i+1 (1-based)

# For each player, compute the answer
for i in range(N):
    # Find max_other
    max_other = max(current_totals[j] for j in range(N) if j != i)
    current_total = current_totals[i]
    diff = max_other - current_total
    if diff < 0:
        print(0)
    else:
        target_sum = diff + 1
        # Find unsolved scores
        unsolved_scores = []
        for j in range(M):
            if S[i][j] == 'x':
                unsolved_scores.append(A[j])
        # Sort descending
        unsolved_scores.sort(reverse=True)
        # Cumulative sum to find min k
        cum_sum = 0
        for k in range(len(unsolved_scores)):
            cum_sum += unsolved_scores[k]
            if cum_sum >= target_sum:
                print(k + 1)  # k starts from 0, so k+1 is the number
                break  # No need to continue