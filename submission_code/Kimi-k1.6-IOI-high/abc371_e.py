n = int(input())
A = list(map(int, input().split()))
last_occurrence = dict()
result = 0
for k in range(n):
    a = A[k]
    prev = last_occurrence.get(a, -1)
    contribution = (k - prev) * (n - k)
    result += contribution
    last_occurrence[a] = k
print(result)