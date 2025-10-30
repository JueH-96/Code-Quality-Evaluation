# YOUR CODE HERE
n, s, m, l = map(int, input().split())

ans = float('inf')

for i in range(n // 6 + 2):
    for j in range(n // 8 + 2):
        for k in range(n // 12 + 2):
            total_eggs = i * 6 + j * 8 + k * 12
            if total_eggs >= n:
                ans = min(ans, i * s + j * m + k * l)

print(ans)