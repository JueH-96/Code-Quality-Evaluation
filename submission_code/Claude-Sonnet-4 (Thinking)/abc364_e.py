n, x, y = map(int, input().split())
dishes = []
for _ in range(n):
    a, b = map(int, input().split())
    dishes.append((a, b))

dp = {(0, 0): 0}

for a, b in dishes:
    new_dp = dp.copy()
    for (s, t), count in dp.items():
        if s <= x and t <= y:  # Can still eat from this state
            new_s, new_t = s + a, t + b
            if (new_s, new_t) not in new_dp:
                new_dp[(new_s, new_t)] = count + 1
            else:
                new_dp[(new_s, new_t)] = max(new_dp[(new_s, new_t)], count + 1)
    dp = new_dp

print(max(dp.values()))