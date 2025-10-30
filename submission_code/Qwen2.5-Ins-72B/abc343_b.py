# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = [[0] * N for _ in range(N)]
    
    idx = 1
    for i in range(N):
        for j in range(N):
            A[i][j] = int(data[idx])
            idx += 1
    
    for i in range(N):
        connected = []
        for j in range(N):
            if A[i][j] == 1:
                connected.append(j + 1)
        print(' '.join(map(str, connected)))

if __name__ == "__main__":
    main()