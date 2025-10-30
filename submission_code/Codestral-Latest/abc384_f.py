import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

def f(x):
    while x % 2 == 0:
        x //= 2
    return x

result = 0
for i in range(N):
    for j in range(i, N):
        result += f(A[i] + A[j])

print(result)