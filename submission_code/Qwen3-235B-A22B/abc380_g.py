class FenwickTree:
    def __init__(self, max_val):
        self.max_val = max_val
        self.tree = [0] * (self.max_val + 2)

    def update(self, x, delta):
        while x <= self.max_val:
            self.tree[x] += delta
            x += x & -x

    def query(self, x):
        res = 0
        while x > 0:
            res += self.tree[x]
            x -= x & -x
        return res

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    K = int(data[ptr])
    ptr += 1
    P = list(map(int, data[ptr:ptr+N]))
    MOD = 998244353

    # Compute original inversion count
    class OriginalBIT:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)

        def update(self, x, delta):
            while x <= self.n:
                self.tree[x] += delta
                x += x & -x

        def query(self, x):
            res = 0
            while x > 0:
                res += self.tree[x]
                x -= x & -x
            return res

    obit = OriginalBIT(N)
    original_inv = 0
    for i in reversed(range(N)):
        x = P[i]
        original_inv += obit.query(x - 1)
        obit.update(x, 1)
    original_inv %= MOD

    T_val = N - K + 1
    if T_val <= 0:
        s2 = 0
    else:
        ft = FenwickTree(N)
        current_inversion = 0
        current_size = 0
        sum_s2 = 0

        # Initialize first window
        for i in range(K):
            x = P[i]
            elements_greater = current_size - ft.query(x)
            current_inversion += elements_greater
            ft.update(x, 1)
            current_size += 1

        sum_s2 = current_inversion % MOD

        # Process remaining T_val - 1 windows
        for s in range(1, T_val):
            # Remove leftmost element x = P[s-1]
            x = P[s - 1]
            q_x = ft.query(x)
            elements_greater = current_size - q_x
            current_inversion -= elements_greater
            ft.update(x, -1)
            current_size -= 1

            # Add new right element y = P[s + K - 1]
            y = P[s + K - 1]
            elements_greater_new = current_size - ft.query(y)
            current_inversion += elements_greater_new
            ft.update(y, 1)
            current_size += 1

            # Add current_inversion to sum_s2
            sum_s2 = (sum_s2 + current_inversion) % MOD

        s2 = sum_s2 % MOD

    # Compute adjustment
    T = T_val
    term1 = (K % MOD) * ((K - 1) % MOD) % MOD
    inv4 = pow(4, MOD - 2, MOD)
    term1 = term1 * inv4 % MOD

    inv_T = pow(T, MOD - 2, MOD) if T != 0 else 0
    term2 = s2 * inv_T % MOD

    adjustment = (term1 - term2 + MOD) % MOD
    expected = (original_inv + adjustment) % MOD
    print(expected)

if __name__ == '__main__':
    main()