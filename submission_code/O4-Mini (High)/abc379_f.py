import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    H = [0] + list(map(int, input().split()))

    # Compute P[j] = index of the nearest building to the left of j that is taller than H[j],
    # or 0 if none.
    P = [0] * (N + 1)
    stack = []
    for j in range(1, N + 1):
        # Pop all buildings that are not taller than H[j]
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        P[j] = stack[-1] if stack else 0
        stack.append(j)

    # Group queries by their r value
    queries_at = [[] for _ in range(N + 1)]
    answers = [0] * Q
    for qi in range(Q):
        l, r = map(int, input().split())
        queries_at[r].append((qi, l))

    # Fenwick / BIT on P-values (0..N), we store counts of added j's by P[j]+1 index.
    size = N + 1
    BIT = [0] * (size + 1)

    def bit_add(i, v):
        while i <= size:
            BIT[i] += v
            i += i & -i

    def bit_sum(i):
        s = 0
        while i > 0:
            s += BIT[i]
            i -= i & -i
        return s

    # Process r from N down to 1.  When at r, we have already added all j > r.
    # To add j>r, we at each step add j = r+1 (if within range).
    for r in range(N, 0, -1):
        j = r + 1
        if j <= N:
            # Insert building j into our BIT, keyed by P[j]+1
            idx = P[j] + 1
            bit_add(idx, 1)
        # Answer queries with this r
        for qi, l in queries_at[r]:
            # Count how many j>r have P[j] <= l
            # P[j] <= l  <=>  P[j]+1 <= l+1
            answers[qi] = bit_sum(l + 1)

    # Output answers
    out = sys.stdout
    out.write("
".join(map(str, answers)))
    out.write("
")

if __name__ == "__main__":
    main()