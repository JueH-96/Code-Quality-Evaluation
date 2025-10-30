n, q = map(int, input().split())
treatments = list(map(int, input().split()))
counts = [0] * n
for t in treatments:
    counts[t-1] += 1
print(sum(1 for x in counts if x % 2 == 0))