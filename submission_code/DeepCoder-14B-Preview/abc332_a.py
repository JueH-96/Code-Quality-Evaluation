n, s, k = map(int, input().split())
total = 0
for _ in range(n):
    p, q = map(int, input().split())
    total += p * q
print(total + (k if total < s else 0))