n = int(input())
h_list = list(map(int, input().split()))

damage_table = [
    [0, 3, 4],
    [0, 1, 2],
    [0, 1, 4],
]

t = 0
for h in h_list:
    r = (t + 1) % 3
    min_candidate = float('inf')
    for s in range(3):
        damage = damage_table[r][s]
        required_m = max(0, (h - damage + 4) // 5)
        candidate = required_m * 3 + s
        if candidate < min_candidate:
            min_candidate = candidate
    t += min_candidate
print(t)