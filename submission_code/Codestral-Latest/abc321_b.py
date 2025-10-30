import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
scores = list(map(int, data[2:]))

scores.sort()
target_sum = sum(scores[1:-1])

if target_sum >= X:
    print(0)
else:
    needed_score = X - target_sum
    if needed_score > 100:
        print(-1)
    else:
        print(needed_score)