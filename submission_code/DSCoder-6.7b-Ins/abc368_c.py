import sys

N = int(sys.stdin.readline().strip())
healths = list(map(int, sys.stdin.readline().strip().split()))

T = 0
while any(h > 0 for h in healths):
    min_health = min(healths)
    for i in range(N):
        if healths[i] > 0:
            if T % 3 == 0:
                healths[i] -= 3
            else:
                healths[i] -= 1
            T += 1
        if healths[i] <= 0:
            healths[i] = 0

print(T)