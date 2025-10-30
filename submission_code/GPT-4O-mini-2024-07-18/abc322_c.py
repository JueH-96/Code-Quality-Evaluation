def fireworks_days(N, M, A):
    results = []
    j = 0  # Pointer for fireworks days

    for i in range(1, N + 1):
        # Move the pointer to the next fireworks day that is >= i
        while j < M and A[j] < i:
            j += 1
        
        if j < M and A[j] == i:
            results.append(0)  # Fireworks on the same day
        elif j < M:
            results.append(A[j] - i)  # Days until the next fireworks
        else:
            results.append(N - i)  # This case won't happen due to constraints

    return results

import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

results = fireworks_days(N, M, A)

for result in results:
    print(result)