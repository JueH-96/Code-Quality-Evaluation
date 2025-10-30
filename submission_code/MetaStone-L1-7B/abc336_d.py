def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    left_max = [0] * N
    for i in range(N):
        max_j = 0
        for j in range(1, N):
            if i - j < 0:
                break
            if A[i - j] >= j:
                max_j += 1
            else:
                break
        left_max[i] = max_j
    
    right_max = [0] * N
    for i in range(N):
        max_j = 0
        for j in range(1, N):
            if i + j >= N:
                break
            if A[i + j] >= j:
                max_j += 1
            else:
                break
        right_max[i] = max_j
    
    max_k = 0
    for i in range(N):
        current_k = min(left_max[i], right_max[i]) + 1
        if current_k > max_k:
            max_k = current_k
    
    print(max_k)

if __name__ == '__main__':
    main()