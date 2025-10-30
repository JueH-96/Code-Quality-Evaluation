# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    D = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:]))
    
    A.sort()
    B.sort()
    
    max_sum = -1
    j = 0
    
    for a in A:
        while j < M and B[j] < a - D:
            j += 1
        if j < M and abs(B[j] - a) <= D:
            max_sum = max(max_sum, a + B[j])
        if j + 1 < M and abs(B[j + 1] - a) <= D:
            max_sum = max(max_sum, a + B[j + 1])
    
    print(max_sum)

if __name__ == "__main__":
    solve()