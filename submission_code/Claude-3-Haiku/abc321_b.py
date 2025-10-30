# Read the input
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Sort the scores in ascending order
A.sort()

# Calculate the minimum score needed in round N
total_score = sum(A[1:-1])
if total_score + 100 < X:
    print(-1)
else:
    for score in range(101):
        new_scores = A[:-1] + [score]
        new_total_score = sum(sorted(new_scores)[1:-1])
        if new_total_score >= X:
            print(score)
            break