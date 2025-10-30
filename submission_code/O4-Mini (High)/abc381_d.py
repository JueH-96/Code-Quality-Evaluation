import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    ans_pairs = 0
    # We handle two parities of starting positions (0-based): p=0 or p=1
    # For each parity p, consider the sequence of possible pair-starts
    # at indices i = p, p+2, p+4, ... where i+1 < N.
    # We build a sliding window over these "pair slots" j = 0..M-1,
    # checking A[i]==A[i+1] (Q[j]==1) and enforcing distinctness of A[i].
    for p in (0, 1):
        M = (N - p) // 2
        if M <= 0:
            continue
        # last_occ[x] = last index j where value x was used in the window
        last_occ = [-1] * (N + 1)
        L = 0  # window left pointer in slots j
        for j in range(M):
            i = p + 2 * j
            # Check if A[i], A[i+1] form a valid pair
            if A[i] != A[i + 1]:
                # Invalid pair breaks any window
                L = j + 1
                continue
            x = A[i]
            prev = last_occ[x]
            # If x was used in the current window, move L past it
            if prev >= L:
                L = prev + 1
            last_occ[x] = j
            # window size in slots is (j - L + 1)
            curr_len = j - L + 1
            if curr_len > ans_pairs:
                ans_pairs = curr_len

    # Each slot corresponds to a pair of length 2
    print(ans_pairs * 2)

# Don't forget to call main!
if __name__ == "__main__":
    main()