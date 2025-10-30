# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

MOD = 10**8

def f(x, y):
    return (x + y) % MOD

total = 0
for i in range(N):
    for j in range(i+1, N):
        total += f(A[i], A[j])

print(total)