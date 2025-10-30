import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    for i in range(N):
        prefix_1[i+1] = prefix_1[i] + (1 if S[i] == '1' else 0)
        prefix_2[i+1] = prefix_2[i] + (1 if S[i] == '2' else 0)

    slashes = []
    for i in range(N):
        if S[i] == '/':
            slashes.append(i)

    if slashes:
        values = [prefix_1[i] - prefix_2[i+1] for i in slashes]
        n = len(values)
        log_table = [0] * (n + 1)
        for i in range(2, n + 1):
            log_table[i] = log_table[i // 2] + 1
        k_max = log_table[n] if n > 0 else 0
        st = [values.copy()]
        for k in range(1, k_max + 1):
            prev = st[k-1]
            curr_len = n - (1 << k) + 1
            curr = [0] * curr_len
            for i in range(curr_len):
                curr[i] = max(prev[i], prev[i + (1 << (k-1))])
            st.append(curr)
    else:
        st = []

    for _ in range(Q):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        L_prime = L - 1
        R_prime = R - 1

        if not slashes:
            print(0)
            continue

        left = bisect.bisect_left(slashes, L_prime)
        right_idx = bisect.bisect_right(slashes, R_prime) - 1

        if left > right_idx:
            print(0)
            continue

        l = right_idx - left + 1
        k = log_table[l] if l > 0 else 0
        max_val = max(st[k][left], st[k][right_idx - (1 << k) + 1])

        max_ab = max_val + (prefix_2[R_prime + 1] - prefix_1[L_prime])
        if max_ab % 2 == 0:
            current_max = max_ab + 1
        else:
            current_max = max_ab

        print(current_max)

if __name__ == '__main__':
    main()