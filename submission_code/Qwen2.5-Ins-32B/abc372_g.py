# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    result = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = []
        B = []
        C = []
        for i in range(N):
            A.append(int(data[index]))
            index += 1
            B.append(int(data[index]))
            index += 1
            C.append(int(data[index]))
            index += 1
        
        max_x = min((C[i] - 1) // A[i] for i in range(N))
        max_y = min((C[i] - 1) // B[i] for i in range(N))
        
        count = 0
        for x in range(1, max_x + 1):
            for y in range(1, max_y + 1):
                if all(A[i] * x + B[i] * y < C[i] for i in range(N)):
                    count += 1
        
        result.append(str(count))
    
    sys.stdout.write("
".join(result))

solve()