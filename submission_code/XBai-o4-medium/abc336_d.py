def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if N == 0:
        print(0)
        return
    
    left_max = [0] * N
    left_max[0] = A[0]
    for i in range(1, N):
        left_max[i] = min(A[i], left_max[i-1] + 1)
    
    right_max = [0] * N
    right_max[N-1] = A[N-1]
    for i in range(N-2, -1, -1):
        right_max[i] = min(A[i], right_max[i+1] + 1)
    
    max_k = 0
    for c in range(N):
        allowed_m = min(c + 1, N - c)
        current_m = min(left_max[c], right_max[c], allowed_m)
        if current_m > max_k:
            max_k = current_m
    
    print(max_k)

if __name__ == "__main__":
    main()