import sys

def main():
    input = sys.stdin.readline
    n, q = map(int, input().split())
    s = input().strip()
    # If length is 1, any substring of length 1 is always good.
    if n == 1:
        w = sys.stdout.write
        for _ in range(q):
            parts = input().split()
            if parts[0] == '2':
                w("Yes
")
        return

    # We maintain an array A of length n-1 where
    # A[i] = 1 if s[i] == s[i+1], else 0.
    # A substring [L..R] is "good" iff sum of A[L..R-1] == 0.
    size = n - 1
    A = [0] * (size + 1)   # 1-based indexing
    bit = [0] * (size + 1) # Fenwick tree array, 1..size

    # Build A and initialize Fenwick tree in O(n).
    for i in range(1, size + 1):
        if s[i-1] == s[i]:
            A[i] = 1
            bit[i] = 1
    # Build fenwick sums in O(n)
    for i in range(1, size + 1):
        j = i + (i & -i)
        if j <= size:
            bit[j] += bit[i]

    write = sys.stdout.write

    # Process queries
    for _ in range(q):
        parts = input().split()
        t = parts[0]
        L = int(parts[1])
        R = int(parts[2])

        if t == '1':
            # Flip s[L..R] toggles A[L-1] and A[R] if in range.
            if L > 1:
                idx = L - 1
                old = A[idx]
                new = 1 - old
                A[idx] = new
                delta = new - old
                j = idx
                while j <= size:
                    bit[j] += delta
                    j += j & -j
            if R < n:
                idx = R
                old = A[idx]
                new = 1 - old
                A[idx] = new
                delta = new - old
                j = idx
                while j <= size:
                    bit[j] += delta
                    j += j & -j

        else:
            # Query type 2: check if substring [L..R] is good.
            if L == R:
                # Single char is always good
                write("Yes
")
            else:
                # sum A[L..R-1] via two Fenwick prefix sums
                left = L
                right = R - 1
                ssum = 0
                j = right
                while j > 0:
                    ssum += bit[j]
                    j -= j & -j
                j = left - 1
                while j > 0:
                    ssum -= bit[j]
                    j -= j & -j

                write("Yes
" if ssum == 0 else "No
")

if __name__ == "__main__":
    main()