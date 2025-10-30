import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    max_pairs = 0

    # We examine two possible "parities" of pairing:
    # parity = 0 means pairs (A[0],A[1]), (A[2],A[3]), ...
    # parity = 1 means pairs (A[1],A[2]), (A[3],A[4]), ...
    # For each parity, build an array v where
    #   v[j] = value of the j-th pair if A[I]==A[I+1], else -1.
    # Then we want the longest subarray of v without -1 and with all values distinct.
    for parity in (0, 1):
        # Build v
        v = []
        # I goes from parity to N-2 in steps of 2
        # Each I defines a candidate pair (A[I], A[I+1])
        for I in range(parity, N - 1, 2):
            if A[I] == A[I + 1]:
                v.append(A[I])
            else:
                v.append(-1)

        # Now sliding window on v to find longest window without -1 and with unique values
        L = 0
        seen = set()
        for R, val in enumerate(v):
            if val == -1:
                # Invalidate whole window
                seen.clear()
                L = R + 1
            else:
                # If val already in window, move L up until it's removed
                while val in seen:
                    seen.remove(v[L])
                    L += 1
                seen.add(val)
                # window length in pairs is R-L+1
                length_pairs = R - L + 1
                if length_pairs > max_pairs:
                    max_pairs = length_pairs

    # Each "pair" contributes 2 to the sequence length
    print(max_pairs * 2)

if __name__ == "__main__":
    main()