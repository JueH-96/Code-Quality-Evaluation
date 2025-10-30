def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Precompute consecutive '1's ending at each position (from the left)
    left_ones = [0] * N
    if S[0] == '1':
        left_ones[0] = 1
    for i in range(1, N):
        if S[i] == '1':
            left_ones[i] = left_ones[i - 1] + 1
        else:
            left_ones[i] = 0

    # Precompute consecutive '2's starting at each position (from the right)
    right_twos = [0] * N
    if S[N - 1] == '2':
        right_twos[N - 1] = 1
    for i in range(N - 2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i + 1] + 1
        else:
            right_twos[i] = 0

    # For each '/', calculate the maximum possible "1^k / 2^k" substring
    # by pairing consecutive '1's to the left and '2's to the right
    ans = 0
    for i in range(N):
        if S[i] == '/':
            c1 = left_ones[i - 1] if i > 0 else 0
            c2 = right_twos[i + 1] if i < N - 1 else 0
            ans = max(ans, 2 * min(c1, c2) + 1)

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()