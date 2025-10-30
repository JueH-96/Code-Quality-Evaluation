# YOUR CODE HERE
import sys
MOD = 998244353

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def get_total(self):
        return self.query(self.n)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr + N]))
    ptr += N

    # Compute original inversion count
    original_inversion = 0
    ft = FenwickTree(N)
    for i in reversed(range(N)):
        val = P[i]
        original_inversion += ft.query(val - 1)
        ft.update(val, 1)

    # Compute SUM
    if K == 1:
        sum_val = 0
    else:
        T = N - K + 1
        if T == 0:
            sum_val = 0
        else:
            sum_val = 0
            ft_window = FenwickTree(N)
            current_inv = 0
            window_size = K
            # Initialize first window
            for i in range(window_size - 1, -1, -1):
                val = P[i]
                current_inv += ft_window.query(val - 1)
                ft_window.update(val, 1)
            contribution = (window_size * (window_size - 1) // 2) - 2 * current_inv
            sum_val += contribution

            # Process remaining windows
            for window_start in range(1, T):
                left_val = P[window_start - 1]
                right_val = P[window_start + K - 1]

                # Remove left_val
                cnt_less = ft_window.query(left_val - 1)
                current_inv -= cnt_less
                ft_window.update(left_val, -1)

                # Add right_val
                total_elements = ft_window.get_total()
                q = ft_window.query(right_val)
                cnt_greater = total_elements - q
                current_inv += cnt_greater
                ft_window.update(right_val, 1)

                # compute contribution
                contribution = (K * (K - 1) // 2) - 2 * current_inv
                sum_val += contribution

    # Compute delta
    T = N - K + 1
    if T == 0:
        delta = 0
    else:
        denominator = 2 * T
        numerator = sum_val
        inv_denominator = pow(denominator, MOD - 2, MOD)
        delta = (numerator % MOD) * inv_denominator % MOD

    # Compute expected value
    expected = (original_inversion % MOD + delta) % MOD
    print(expected)

if __name__ == '__main__':
    main()