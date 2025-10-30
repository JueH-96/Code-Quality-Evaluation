def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    # Precompute consecutive '1's ending at each position.
    # left_ones[i] is the length of consecutive '1's ending at index i (0-indexed).
    left_ones = [0] * N
    if N > 0:
        left_ones[0] = 1 if S[0] == '1' else 0
    for i in range(1, N):
        if S[i] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0

    # Precompute consecutive '2's starting at each position.
    # right_twos[i] is the length of consecutive '2's starting at index i.
    right_twos = [0] * N
    if N > 0:
        right_twos[N-1] = 1 if S[N-1] == '2' else 0
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0

    # An "11/22 string" is defined as "1"*k + "/" + "2"*k for some k>=0.
    # Thus, for every index j with S[j]=='/', the maximum k possible is the minimum of:
    #    - The number of consecutive '1's immediately before j.
    #    - The number of consecutive '2's immediately after j.
    # The maximum length of such a substring centered at j is 2*k + 1.
    max_length = 0
    for j in range(N):
        if S[j] == '/':
            left_count = left_ones[j-1] if j - 1 >= 0 else 0
            right_count = right_twos[j+1] if j + 1 < N else 0
            k = min(left_count, right_count)
            current_length = 2 * k + 1
            if current_length > max_length:
                max_length = current_length

    # Output the result.
    sys.stdout.write(str(max_length))

if __name__ == '__main__':
    main()