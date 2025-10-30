import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    
    # Compute C[j]: number of times B[j] appears in B
    C = [0] * N
    for b in B:
        C[b] += 1
    
    # Initialize difference array
    diff = [0] * (N + 1)
    
    for i in range(M):
        a = B[i]
        K_i = A[a]
        if K_i == 0:
            continue
        b = (a + K_i - 1) % N
        
        if a <= b:
            # Single interval [a, b]
            diff[a] += 1
            if b + 1 < N:
                diff[b + 1] -= 1
            else:
                if 0 < N:
                    diff[0] -= 1
        else:
            # Two intervals: [a, N-1] and [0, b]
            diff[a] += 1
            if N < b + 1:
                diff[N] -= 1
            else:
                if 0 < N:
                    diff[0] -= 1
            diff[0] += 1
            if b + 1 < N:
                diff[b + 1] -= 1
            else:
                if 0 < N:
                    diff[N] -= 1
    
    # Compute coverage array
    coverage = [0] * N
    current = 0
    for j in range(N):
        current += diff[j]
        coverage[j] = current
    
    # Compute final A
    res = [0] * N
    for j in range(N):
        if C[j] > 0:
            res[j] = coverage[j]
        else:
            res[j] = A[j] + coverage[j]
    
    print(' '.join(map(str, res)))

if __name__ == '__main__':
    main()