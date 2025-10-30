import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    if n == 0:
        print()
        return
    # Compute an upper bound for T = S_j0 + j = (A_j + gifts_j) + j.
    # gifts_j <= j-1, so T <= A_j + (j-1) + j = A_j + 2*j -1
    maxA = max(A)
    size = maxA + 2*n + 5
    # Fenwick tree for counts of T_i
    tree = [0] * (size)
    # Fenwick add and prefix sum
    def bit_add(i):
        # add 1 at index i
        while i < size:
            tree[i] += 1
            i += i & -i
    def bit_sum(i):
        # sum of [1..i]
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    S0 = [0] * n  # store S_j0 = A_j + gifts_j
    # process events j = 1..n
    for j in range(1, n+1):
        if j == 1:
            gifts = 0
        else:
            # count of previous T_i >= j = (j-1) - count of T_i <= j-1
            less_eq = bit_sum(j-1)
            gifts = (j-1) - less_eq
        s = A[j-1] + gifts
        S0[j-1] = s
        # compute T_j and add to Fenwick
        T = s + j
        # ensure T>=1
        if T < 1:
            T = 1
        # T should be within our size
        bit_add(T)
    # Now compute final B_j = max(0, S0_j - (n - j))
    out = []
    for j in range(1, n+1):
        bj = S0[j-1] - (n - j)
        if bj < 0:
            bj = 0
        out.append(str(bj))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()