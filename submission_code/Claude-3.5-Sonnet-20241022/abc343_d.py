N, T = map(int, input().split())
scores = [0] * (N+1)  # scores[i] is score of player i

for i in range(T):
    a, b = map(int, input().split())
    scores[a] += b
    # Count unique scores at this point
    unique = len(set(scores[1:]))  # exclude scores[0]
    print(unique)