n, q = map(int, input().split())
freq = [0] * (n + 1)  # Using 1-based indexing

for _ in range(q):
    t = int(input())
    freq[t] += 1

result = 0
for i in range(1, n + 1):
    if freq[i] % 2 == 0:
        result += 1

print(result)