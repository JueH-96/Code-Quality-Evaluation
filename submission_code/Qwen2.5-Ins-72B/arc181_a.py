# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

def min_operations_to_sort(N, P):
    operations = 0
    for i in range(1, N + 1):
        if P[i - 1] != i:
            operations += 1
            k = P.index(i) + 1
            if k >= 2:
                P[:k-1] = sorted(P[:k-1])
            if k <= N - 1:
                P[k:] = sorted(P[k:])
    return operations

T = int(data[0])
index = 1
results = []
for _ in range(T):
    N = int(data[index])
    index += 1
    P = list(map(int, data[index].split()))
    index += 1
    results.append(min_operations_to_sort(N, P))

for result in results:
    print(result)