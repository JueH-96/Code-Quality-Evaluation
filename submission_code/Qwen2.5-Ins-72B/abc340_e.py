# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:]))
    
    for i in range(M):
        b = B[i]
        c = A[b]
        A[b] = 0
        for j in range(c):
            A[(b + j + 1) % N] += 1
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()