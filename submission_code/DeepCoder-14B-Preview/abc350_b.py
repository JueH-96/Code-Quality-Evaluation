n, q = map(int, input().split())
treatments = list(map(int, input().split()))

freq = [0] * (n + 1)  # Using 1-based indexing

for t in treatments:
    freq[t] += 1

total = 0
for j in range(1, n + 1):
    if freq[j] % 2 == 0:
        total += 1

print(total)