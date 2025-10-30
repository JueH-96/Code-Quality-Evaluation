def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    P = [int(next(it)) for _ in range(n)]
    mod = 998244353
    M = n - K + 1  # number of possible positions for the block

    # BIT (Fenwick tree) for inversion counts.
    class BIT:
        __slots__ = ['n', 'tree']
        def __init__(self, n):
            self.n = n
            self.tree = [0]*(n+1)
        def update(self, i, delta):
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            while i:
                s += self.tree[i]
                i -= i & -i
            return s
        def query_range(self, l, r):
            if l > r:
                return 0
            return self.query(r) - self.query(l-1)
    
    # 1. Compute T = inversion count for P.
    T = 0
    bit_full = BIT(n)
    for i in range(n):
        x = P[i]
        # count how many of the ones already seen (which are in positions before i)
        # are greater than x
        bigger = bit_full.query_range(x+1, n)
        T += bigger
        bit_full.update(x, 1)
    
    # 2. Compute the sum (over every contiguous segment of length K) of inversion counts X.
    # We use a sliding window technique.
    bit_win = BIT(n)
    current_inv = 0
    win_size = 0
    # Build initial window: indices 0 ... K-1.
    for i in range(K):
        x = P[i]
        # When inserting x at the right, the number of inversions added is the number
        # of elements already in the window that are greater than x.
        add_val = win_size - bit_win.query(x)
        current_inv += add_val
        win_size += 1
        bit_win.update(x, 1)
    S_sum = current_inv

    # Slide the window from starting index L = 0 to L = M-1 (which corresponds to positions 1 through M).
    for L in range(0, M - 1):
        # Remove leftmost element (which is at index L).
        x_remove = P[L]
        # Because the window is maintained in order (we always remove the oldest element),
        # all the remaining elements come after the removed element.
        # The number of inversions contributed by the removed value is the number
        # of remaining elements smaller than it.
        rem_contrib = bit_win.query(x_remove - 1)
        current_inv -= rem_contrib
        bit_win.update(x_remove, -1)
        win_size -= 1
        # Add the new element at index L+K.
        x_add = P[L + K]
        add_contrib = win_size - bit_win.query(x_add)
        current_inv += add_contrib
        bit_win.update(x_add, 1)
        win_size += 1
        S_sum += current_inv

    # 3. Our overall expected inversion count (averaged over the choice of block) is:
    #    E = T + (K*(K-1)/4) - (S_sum / M)
    # Multiply both sides by M:
    #    E * M = T*M + M*(K*(K-1)/4) - S_sum
    # Then the answer we output is (E * M)/M in the field modulo mod.
    #
    # Compute numerator = T*M + M*(K*(K-1)/4) - S_sum, and then multiply by the mod inverse of M.
    inv4 = pow(4, mod-2, mod)
    term1 = (T % mod) * (M % mod) % mod
    term2 = (M % mod) * (((K % mod) * ((K - 1) % mod)) % mod) % mod
    term2 = (term2 * inv4) % mod
    numerator = (term1 + term2 - (S_sum % mod)) % mod
    invM = pow(M, mod-2, mod)
    ans = numerator * invM % mod
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()