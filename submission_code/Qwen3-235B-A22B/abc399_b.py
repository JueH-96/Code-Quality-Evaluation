n = int(input())
p = list(map(int, input().split()))
processed = [False] * n
ranks = [0] * n
r = 1

while not all(processed):
    current_scores = [p[i] for i in range(n) if not processed[i]]
    max_score = max(current_scores)
    same = [i for i in range(n) if not processed[i] and p[i] == max_score]
    for idx in same:
        ranks[idx] = r
    r += len(same)
    for idx in same:
        processed[idx] = True

for rank in ranks:
    print(rank)