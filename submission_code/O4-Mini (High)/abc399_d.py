import sys
from array import array

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    T = int(next(it))
    out_lines = []
    # We'll pack each segment as a single integer:
    #   comp = (pair_id << 19) | pos,
    # using 19 bits for pos (max pos < 2*200k = 400k < 2^19).
    mask = (1 << 19) - 1

    for _ in range(T):
        N = int(next(it))
        size = N * 2

        # Read the 2N seats
        A = [0] * size
        for i in range(size):
            A[i] = int(next(it))

        # Mark which couples are already adjacent (and thus invalid)
        # Use a bytearray to save memory: 1 means "bad".
        isAdj = bytearray(N+1)
        for i in range(size - 1):
            if A[i] == A[i+1]:
                isAdj[A[i]] = 1

        # Build a list of "segment codes" for each adjacent-seat segment
        # that lies between two distinct, non‐self‐adjacent couples.
        mul = N + 1
        arr = []
        for i in range(size - 1):
            a = A[i]
            b = A[i+1]
            if a != b and not isAdj[a] and not isAdj[b]:
                # normalize pair (u,v) with u < v
                if a < b:
                    u, v = a, b
                else:
                    u, v = b, a
                pair_id = u * mul + v
                comp = (pair_id << 19) | i
                arr.append(comp)

        # Sort the segments by (pair_id, position)
        arr.sort()

        # Scan each group of identical pair_id's and check if there
        # exist two segment positions that differ by more than 1.
        ans = 0
        m = len(arr)
        idx = 0
        while idx < m:
            base = arr[idx]
            pid = base >> 19
            p0  = base & mask
            j = idx + 1
            # look for a later segment of the same pair
            while j < m and (arr[j] >> 19) == pid:
                p = arr[j] & mask
                if p - p0 > 1:
                    ans += 1
                    # skip the rest of this pair's segments
                    k = j + 1
                    while k < m and (arr[k] >> 19) == pid:
                        k += 1
                    idx = k
                    break
                j += 1
            else:
                # no qualifying second segment
                idx = j

        out_lines.append(str(ans))

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()