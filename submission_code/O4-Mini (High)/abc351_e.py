import sys
import threading

def main():
    data = sys.stdin
    n = int(data.readline())
    # Separate points by parity of (x+y)
    g0_u = []
    g0_v = []
    g1_u = []
    g1_v = []

    for _ in range(n):
        x, y = map(int, data.readline().split())
        u = x + y
        v = x - y
        if (u & 1) == 0:
            g0_u.append(u)
            g0_v.append(v)
        else:
            g1_u.append(u)
            g1_v.append(v)

    def sum_abs_diff(arr):
        """Return sum_{i<j} |arr[i] - arr[j]|."""
        m = len(arr)
        if m < 2:
            return 0
        arr.sort()
        total = 0
        prefix = 0
        # For each arr[i], the contribution to sum_{j<i} (arr[i] - arr[j]) is
        # arr[i]*i - sum_{j<i} arr[j]
        for i, val in enumerate(arr):
            total += val * i - prefix
            prefix += val
        return total

    ans = 0
    # For each parity group, sum of Chebyshev distances is
    # 1/2 * ( sum |u_i - u_j| + sum |v_i - v_j| )
    for U, V in ((g0_u, g0_v), (g1_u, g1_v)):
        s_u = sum_abs_diff(U)
        s_v = sum_abs_diff(V)
        ans += (s_u + s_v) // 2

    print(ans)

if __name__ == "__main__":
    main()