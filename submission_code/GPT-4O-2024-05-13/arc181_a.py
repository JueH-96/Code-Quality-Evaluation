# YOUR CODE HERE
def min_operations_to_sort_permutation(T, cases):
    results = []
    for case in cases:
        N, P = case
        if P == list(range(1, N+1)):
            results.append(0)
            continue
        
        if P[0] == 1 or P[-1] == N:
            results.append(1)
        elif P[0] == N and P[-1] == 1:
            results.append(3)
        else:
            results.append(2)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
cases = []

for _ in range(T):
    N = int(data[index])
    P = list(map(int, data[index+1:index+1+N]))
    cases.append((N, P))
    index += N + 1

results = min_operations_to_sort_permutation(T, cases)
for result in results:
    print(result)