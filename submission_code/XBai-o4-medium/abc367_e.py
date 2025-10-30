import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    # Make X and A 1-based
    X = [0] + X
    A = [0] + A
    
    log_max = 60
    up = [[0]*(log_max + 1) for _ in range(N+1)]
    
    for i in range(1, N+1):
        up[i][0] = X[i]
    
    for j in range(1, log_max + 1):
        for i in range(1, N+1):
            up[i][j] = up[ up[i][j-1] ][j-1]
    
    res = [0]*(N+1)
    for i in range(1, N+1):
        current = i
        for j in range(log_max, -1, -1):
            if (K >> j) & 1:
                current = up[current][j]
        res[i] = A[current]
    
    print(' '.join(map(str, res[1:N+1])))

if __name__ == "__main__":
    main()