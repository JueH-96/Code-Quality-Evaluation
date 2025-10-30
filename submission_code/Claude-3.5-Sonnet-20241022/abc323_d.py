N = int(input())
slimes = []
for _ in range(N):
    s, c = map(int, input().split())
    slimes.append([s, c])

ans = 0
for s, c in slimes:
    # For each count, we can combine pairs to reduce the count
    # The remaining slimes after combining pairs is c % 2 + c // 2
    # This is equivalent to (c + 1) // 2
    ans += (c + 1) // 2

print(ans)