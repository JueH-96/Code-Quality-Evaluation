import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

def f(x):
    while x % 2 == 0:
        x //= 2
    return x

total_sum = 0

for i in range(N):
    for j in range(i, N):
        total_sum += f(A[i] + A[j])

print(total_sum)