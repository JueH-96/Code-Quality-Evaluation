N, S, K = map(int, input().split())

total = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total += P * Q

if total < S:
    total += K

print(total)