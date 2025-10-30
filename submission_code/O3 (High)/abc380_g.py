import sys

MOD = 998244353


class Fenwick:
    """Fenwick tree for point add / prefix sum (1-indexed)."""
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, idx: int, delta: int) -> None:
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & -idx

    def sum(self, idx: int) -> int:
        res = 0
        while idx:
            res += self.bit[idx]
            idx -= idx & -idx
        return res


def modinv(x: int) -> int:
    return pow(x, MOD - 2, MOD)


def total_inversions(arr, n) -> int:
    bit = Fenwick(n)
    inv = 0
    for i, v in enumerate(arr):
        # number of previously inserted elements greater than v
        inv += i - bit.sum(v)
        bit.add(v, 1)
    return inv


def window_inversion_sum(arr, n, k) -> int:
    """Sum of inversion numbers over all consecutive windows of length k."""
    if k == 1:
        return 0
    bit = Fenwick(n)
    inv_win = 0

    # build first window [0, k)
    for j in range(k):
        v = arr[j]
        inv_win += j - bit.sum(v)
        bit.add(v, 1)

    total = inv_win
    L = n - k + 1
    for s in range(1, L):
        # positions
        left_val = arr[s - 1]
        right_val = arr[s + k - 1]

        # remove left_val
        smaller_cnt = bit.sum(left_val - 1)
        inv_win -= smaller_cnt
        bit.add(left_val, -1)

        # current size is k-1
        curr_size = k - 1
        greater_cnt = curr_size - bit.sum(right_val)
        inv_win += greater_cnt
        bit.add(right_val, 1)

        total += inv_win
    return total


def main() -> None:
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    inv_total = total_inversions(P, N)                       # whole permutation
    sum_windows = window_inversion_sum(P, N, K)              # Σ inversion(window)

    L = N - K + 1                                            # number of windows

    inv_total_mod = inv_total % MOD
    sum_windows_mod = sum_windows % MOD

    invL = modinv(L)                                         # 1 / L (mod)
    inv2 = (MOD + 1) // 2                                    # 1 / 2 (mod)

    # term = (K*(K-1)/2) * (1/2) = K*(K-1)/4
    choose_k2 = K * (K - 1) // 2
    term = (choose_k2 % MOD) * inv2 % MOD

    ans = (inv_total_mod - (sum_windows_mod * invL % MOD) + term) % MOD
    print(ans)


if __name__ == "__main__":
    main()