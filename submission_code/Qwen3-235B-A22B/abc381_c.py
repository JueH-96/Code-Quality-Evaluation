def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Precompute left_ones array
    left_ones = [0] * N
    if S[0] == '1':
        left_ones[0] = 1
    for i in range(1, N):
        if S[i] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0
    
    # Precompute right_twos array
    right_twos = [0] * N
    if S[-1] == '2':
        right_twos[N-1] = 1
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0
    
    max_len = 0
    for i in range(N):
        if S[i] == '/':
            left_count = left_ones[i-1] if i > 0 else 0
            right_count = right_twos[i+1] if i < N-1 else 0
            k = min(left_count, right_count)
            current_len = 2 * k + 1
            if current_len > max_len:
                max_len = current_len
    
    print(max_len)

if __name__ == "__main__":
    main()