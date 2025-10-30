def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    degrees = [0] * (N + 1)
    for _ in range(N - 1):
        u = int(input[ptr])
        v = int(input[ptr + 1])
        ptr += 2
        degrees[u] += 1
        degrees[v] += 1
    
    deg_ge3 = [d for d in degrees[1:N+1] if d >= 3]
    K = len(deg_ge3)
    sum_deg_ge3 = sum(deg_ge3)
    M = (N - sum_deg_ge3 + 2 * K) // 3
    X = M - K
    
    result = deg_ge3 + [2] * X
    result.sort()
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()