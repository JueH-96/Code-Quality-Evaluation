# YOUR CODE HERE

N, X = map(int, input().split())
scores = sorted(map(int, input().split()))

if N == 1:
    print(max(0, X - scores[0]))
else:
    total_score = sum(scores[1:-1])
    if total_score >= X:
        print(total_score)
    else:
        print(-1)