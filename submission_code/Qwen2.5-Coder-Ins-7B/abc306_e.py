import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
Q = int(data[2])

A = [0] * N
index = 3
results = []

for _ in range(Q):
    X = int(data[index]) - 1
    Y = int(data[index + 1])
    index += 2
    
    A[X] = Y
    A.sort(reverse=True)
    result = sum(A[:K])
    results.append(result)

for result in results:
    print(result)