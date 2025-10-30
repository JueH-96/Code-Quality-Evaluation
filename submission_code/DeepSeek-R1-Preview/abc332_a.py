n, s, k = map(int, input().split())
total = 0
for _ in range(n):
    p, q = map(int, input().split())
    total += p * q
shipping = 0 if total >= s else k
print(total + shipping)