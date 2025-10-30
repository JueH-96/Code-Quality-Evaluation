import sys

def main():
    import sys
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Compute prefix_1 and prefix_2 arrays (0-based)
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_1[i] = prefix_1[i-1] + (1 if S[i-1] == '1' else 0)
        prefix_2[i] = prefix_2[i-1] + (1 if S[i-1] == '2' else 0)

    # Compute A_values array (0-based, size N)
    A_values = [-float('inf')] * N
    for i in range(N):
        if S[i] == '/':
            # A is prefix_1[i] (number of 1's up to i-1 (code's i is the current position, so up to i-1 is prefix_1[i])
            # minus prefix_2[i+1] (number of 2's up to i (current position))
            A_values[i] = prefix_1[i] - prefix_2[i+1]
        else:
            A_values[i] = -float('inf')

    # Build sparse table for range maximum query
    n = N
    if n == 0:
        for _ in range(Q):
            print(0)
        return

    k_max = 0
    if n >= 1:
        k_max = (n).bit_length()
        while (1 << (k_max)) <= n:
            k_max += 1
        k_max -= 1

    st = []
    st.append(A_values.copy())
    for k in range(1, k_max + 1):
        prev = st[k-1]
        m = 1 << k
        curr = [0] * (n - m + 1)
        for i in range(n - m + 1):
            mid = i + (1 << (k-1))
            curr[i] = max(prev[i], prev[mid])
        st.append(curr)

    # Process queries
    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())
        a = L - 1
        b = R - 1
        if a > b:
            print(0)
            continue
        length = b - a + 1
        if length == 0:
            print(0)
            continue

        # Compute k for the sparse table query
        k = 0
        while (1 << (k + 1)) <= length:
            k += 1
        max_A = max(st[k][a], st[k][b - (1 << k) + 1])

        if max_A == -float('inf'):
            print(0)
        else:
            B = prefix_2[R] - prefix_1[a]
            max_total = max_A + B
            ans = max_total + (1 if (max_total % 2 == 0) else 0)
            ans = max(ans, 1)
            print(ans)

if __name__ == "__main__":
    main()