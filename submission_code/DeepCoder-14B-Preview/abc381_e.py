import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    q = int(data[idx])
    idx += 1
    s = data[idx]
    idx += 1

    # Precompute prefix_ones and prefix_twos
    prefix_ones = [0] * (n + 1)
    prefix_twos = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_ones[i] = prefix_ones[i-1] + (1 if s[i-1] == '1' else 0)
        prefix_twos[i] = prefix_twos[i-1] + (1 if s[i-1] == '2' else 0)

    # Collect all slashes positions (1-based)
    slashes = []
    for i in range(1, n+1):
        if s[i-1] == '/':
            slashes.append(i)

    if not slashes:
        # No slashes, all queries return 0
        for _ in range(q):
            print(0)
        return

    # Precompute v_list for each slash
    m = len(slashes)
    v_list = []
    for i in slashes:
        ones_before = prefix_ones[i-1]
        twos_after = prefix_twos[i]
        v = ones_before - twos_after
        v_list.append(v)

    # Precompute log table for sparse table
    max_log = max(1, (m.bit_length()))
    log_table = [0] * (m + 1)
    for k in range(2, m + 1):
        log_table[k] = log_table[k // 2] + 1

    # Build sparse table
    k_max = log_table[m] + 1 if m > 0 else 0
    st = []
    st.append(v_list.copy())
    for k in range(1, k_max):
        current = []
        for j in range(m - (1 << k) + 1):
            val = max(st[k-1][j], st[k-1][j + (1 << (k-1))])
            current.append(val)
        st.append(current)

    # Function to get max in range [l, r] (0-based)
    def get_max(l, r):
        if l > r:
            return -float('inf')
        length = r - l + 1
        k = log_table[length]
        if (1 << k) > length:
            k -= 1
        return max(st[k][l], st[k][r - (1 << k) + 1])

    # Process each query
    results = []
    for _ in range(q):
        L = int(data[idx])
        idx += 1
        R = int(data[idx])
        idx += 1

        a = prefix_ones[L-1]
        b = prefix_twos[R]

        # Find the range in slashes
        left = bisect.bisect_left(slashes, L)
        right = bisect.bisect_right(slashes, R)

        if right - left == 0:
            results.append(0)
            continue

        # Query the max in v_list from left to right-1 (inclusive)
        l = left
        r = right - 1
        max_v = get_max(l, r)

        if max_v == -float('inf'):
            results.append(0)
            continue

        max_current = max_v + (b - a + 1)
        if max_current % 2 == 0:
            max_current -= 1
        if max_current < 1:
            max_current = 1  # At least the '/' can be taken
        results.append(max_current)

    # Output all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()