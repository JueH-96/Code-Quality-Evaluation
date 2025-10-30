# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

def solve_case(N, A, B, C):
    max_x = max(C[i] // A[i] for i in range(N))
    max_y = max(C[i] // B[i] for i in range(N))
    
    count = 0
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            if all(A[i] * x + B[i] * y < C[i] for i in range(N)):
                count += 1
    return count

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    index += 1
    A, B, C = [], [], []
    for _ in range(N):
        a, b, c = map(int, data[index].split())
        A.append(a)
        B.append(b)
        C.append(c)
        index += 1
    results.append(solve_case(N, A, B, C))

for result in results:
    print(result)