from collections import defaultdict

# Read input
N = int(input())
A = list(map(int, input().split()))

# Count the frequency of each element in the sequence
freq = defaultdict(list)
for i, a in enumerate(A):
    freq[a].append(i)

# Count the number of triples
count = 0
for i in range(N):
    for j in range(i+1, N):
        if A[i] != A[j]:
            for k in freq[A[i]]:
                if k > j:
                    count += 1

print(count)