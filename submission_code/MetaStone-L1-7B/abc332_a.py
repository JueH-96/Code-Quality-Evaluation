n, s, k = map(int, input().split())
total = 0
for _ in range(n):
    p, q = map(int, input().split())
    total += p * q
if total >= s:
    fee = 0
else:
    fee = k
print(total + fee)