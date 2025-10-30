n, q = map(int, input().split())
treatments = list(map(int, input().split()))
count = [0] * (n + 1)
for t in treatments:
    count[t] += 1
odd = sum(1 for i in range(1, n+1) if count[i] % 2 == 1)
print(n - odd)