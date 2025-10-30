def main():
    import sys
    N, *rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    
    if N == 0:
        print(0)
        return
    
    left_max = [0] * N
    left_max[0] = A[0]
    for i in range(1, N):
        left_max[i] = min(A[i], left_max[i-1] + 1)
    
    right_max = [0] * N
    right_max[-1] = A[-1]
    for i in range(N-2, -1, -1):
        right_max[i] = min(A[i], right_max[i+1] + 1)
    
    max_k = 0
    for i in range(N):
        a = left_max[i]
        b = right_max[i]
        c = i + 1
        d = N - i
        current_min = min(a, b, c, d)
        if current_min > max_k:
            max_k = current_min
    
    print(max_k)

if __name__ == "__main__":
    main()