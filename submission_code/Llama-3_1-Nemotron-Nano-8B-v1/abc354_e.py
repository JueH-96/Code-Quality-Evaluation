n = int(input())
cards = [tuple(map(int, input().split())) for _ in range(n)]

valid_pairs = []
for i in range(n):
    for j in range(i + 1, n):
        if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
            valid_pairs.append((i, j))

masks = list(range(1 << n))
masks.sort(key=lambda x: bin(x).count('1'))

dp = {mask: False for mask in masks}

for mask in masks:
    for i, j in valid_pairs:
        if (mask & (1 << i)) and (mask & (1 << j)):
            new_mask = mask ^ ((1 << i) | (1 << j))
            if not dp[new_mask]:
                dp[mask] = True
                break

full_mask = (1 << n) - 1
print("Takahashi" if dp[full_mask] else "Aoki")