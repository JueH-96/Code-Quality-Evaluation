# YOUR CODE HERE
N = int(input())
H = [0] + list(map(int, input().split()))  # H[i] is the barrier for flow into position i

A = [0] * (N + 1)
result = [0] * (N + 1)  # result[i] stores when A[i] first becomes positive
operations = 0
found = 0

while found < N:
    operations += 1
    A[0] += 1
    
    for i in range(1, N + 1):
        if A[i-1] > A[i] and A[i-1] > H[i]:
            A[i-1] -= 1
            A[i] += 1
            if result[i] == 0 and A[i] > 0:
                result[i] = operations
                found += 1

print(' '.join(str(result[i]) for i in range(1, N + 1)))