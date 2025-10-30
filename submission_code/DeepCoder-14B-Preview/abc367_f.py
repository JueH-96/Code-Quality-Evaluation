import math

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    A = list(map(int, input[ptr:ptr+N]))
    
    if K == 0:
        print(' '.join(map(str, A)))
        return
    
    max_level = math.floor(math.log2(K)) + 1
    up = [[0] * N for _ in range(max_level)]
    
    for i in range(N):
        up[0][i] = X[i] - 1  # Convert to zero-based index
    
    for j in range(1, max_level):
        for i in range(N):
            up[j][i] = up[j-1][up[j-1][i]]
    
    def get_kth(i, K):
        res = i
        for j in reversed(range(max_level)):
            if K & (1 << j):
                res = up[j][res]
        return res
    
    final = []
    for i in range(N):
        pos = get_kth(i, K)
        final.append(str(A[pos]))
    
    print(' '.join(final))

if __name__ == "__main__":
    main()