# YOUR CODE HERE
def can_make_identical(T, cases):
    results = []
    for case in cases:
        N, K, A, B = case
        if sorted(A) == sorted(B):
            results.append("Yes")
        else:
            results.append("No")
    return results

import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
cases = []

for _ in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    A = list(map(int, data[index + 2:index + 2 + N]))
    B = list(map(int, data[index + 2 + N:index + 2 + 2 * N]))
    cases.append((N, K, A, B))
    index += 2 + 2 * N

results = can_make_identical(T, cases)
for result in results:
    print(result)