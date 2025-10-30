# N T = map(int, input().split())
N, T = 10, 10
# A B = [list(map(int, input().split())) for _ in range(T)]
A_B = [[7, 2620], [9, 2620], [8, 3375], [1, 3375], [6, 1395], [5, 1395], [6, 2923], [10, 3375], [9, 5929], [5, 1225]]

from collections import defaultdict

score_changes = defaultdict(int)
for a, b in A_B:
    score_changes[a] += b

times = sorted(score_changes.keys())
last_score = 0
last_time = 0
score_counts = defaultdict(int)
result = []

for i in range(T):
    time = i + 1
    while times and times[0] <= time:
        score = score_changes[times[0]]
        score_counts[score] += 1
        if score_counts[score] == 1:
            last_score = score
        times.pop(0)
    result.append(len(score_counts) - (last_score == 0))

for r in result:
    print(r)