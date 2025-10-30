n = int(input())
A = list(map(int, input().split()))
conversions = []
for _ in range(n-1):
    s, t = map(int, input().split())
    conversions.append((s, t))

cur = A[0]
for i in range(n-1):
    s, t = conversions[i]
    k = cur // s
    cur = A[i+1] + k * t

print(cur)