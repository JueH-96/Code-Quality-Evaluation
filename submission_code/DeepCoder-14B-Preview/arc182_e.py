# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

total = 0
for a, b in zip(A, B):
    diff = (b - a) % M
    total += min(diff, M - diff)

print(total)