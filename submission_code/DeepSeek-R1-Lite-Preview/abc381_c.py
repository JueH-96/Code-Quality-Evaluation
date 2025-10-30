def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]

    # Precompute left_ones
    left_ones = [0] * N
    for i in range(1, N):
        if S[i-1] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0

    # Precompute right_twos
    right_twos = [0] * N
    for i in range(N-2, -1, -1):
        if S[i+1] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0

    max_k = 0
    for i in range(N):
        if S[i] == '/':
            k = min(left_ones[i], right_twos[i])
            if k > max_k:
                max_k = k

    # Calculate the maximum length
    max_length = 2 * max_k + 1
    print(max_length)

if __name__ == "__main__":
    main()