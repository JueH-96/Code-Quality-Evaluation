n = int(input())
a = list(map(int, input().split()))

r = [0] * n  # r[i] = gifts received by alien i (0-indexed)

for i in range(1, n):
    for j in range(i):
        if a[j] + r[j] >= i - j:
            r[i] += 1

# Calculate final stones
final_stones = []
for i in range(n):
    # Final = initial + received - given
    # Alien i gives to aliens i+1, i+2, ..., n-1 (that's n-1-i aliens)
    # But only while they have stones: min(a[i] + r[i], n-1-i)
    final_stones.append(max(0, a[i] + r[i] - (n - 1 - i)))

print(' '.join(map(str, final_stones)))