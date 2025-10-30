def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    x_list = list(map(int, input[ptr:ptr+Q]))
    ptr += Q

    # Initialize T_j for each j
    T = [[] for _ in range(N+1)]  # T[0] unused
    for idx, x in enumerate(x_list):
        T[x].append(idx)
    
    # Compute s and count
    count = [0] * (N + 1)
    current_size = 0
    s = []
    for x in x_list:
        if count[x] % 2 == 0:
            current_size += 1
        else:
            current_size -= 1
        s.append(current_size)
        count[x] += 1
    
    # Compute prefix sum
    prefix_sum = [0] * (Q + 1)
    for i in range(Q):
        prefix_sum[i+1] = prefix_sum[i] + s[i]
    
    # Compute A
    A = [0] * N
    for j in range(1, N+1):
        m = len(T[j])
        sum_j = 0
        for i in range(0, m, 2):
            a = T[j][i]
            if i + 1 < m:
                b = T[j][i+1] - 1
            else:
                b = Q - 1
            if a > b:
                continue
            sum_j += prefix_sum[b+1] - prefix_sum[a]
        A[j-1] = sum_j
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()