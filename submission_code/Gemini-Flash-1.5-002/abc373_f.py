# YOUR CODE HERE
N, W = map(int, input().split())
items = []
for i in range(N):
    w, v = map(int, input().split())
    items.append((w, v))

dp = {}
dp[0] = 0

for w, v in items:
    new_dp = {}
    for weight, happiness in dp.items():
        new_dp[weight] = happiness
        k = 1
        while True:
            new_weight = weight + k * w
            if new_weight > W:
                break
            new_happiness = happiness + k * v - k * k
            if new_weight not in new_dp or new_dp[new_weight] < new_happiness:
                new_dp[new_weight] = new_happiness
            k += 1
    dp = new_dp

max_happiness = 0
for happiness in dp.values():
    max_happiness = max(max_happiness, happiness)

print(max_happiness)