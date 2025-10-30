def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Precompute how many consecutive '1's end at each position (from left to right)
    left_ones = [0]*N
    if S[0] == '1':
        left_ones[0] = 1
    for i in range(1, N):
        if S[i] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0

    # Precompute how many consecutive '2's start at each position (from right to left)
    right_twos = [0]*N
    if S[N-1] == '2':
        right_twos[N-1] = 1
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0

    # For every position that has '/', compute the maximum k
    # where we can form 1^k / 2^k, which has length 2k+1.
    # k <= number_of_consecutive_1s_to_the_left, and
    # k <= number_of_consecutive_2s_to_the_right.
    # We'll track the maximum length found.
    ans = 1  # at least we can always take just '/'
    for i, ch in enumerate(S):
        if ch == '/':
            # how many 1's to the left?
            left_count = left_ones[i-1] if i-1 >= 0 else 0
            # how many 2's to the right?
            right_count = right_twos[i+1] if i+1 < N else 0
            k = min(left_count, right_count)
            length = 2*k + 1
            if length > ans:
                ans = length

    print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()