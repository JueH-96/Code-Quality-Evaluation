import sys
import threading

def main():
    import sys

    data = sys.stdin
    n = int(data.readline())
    # We partition points by parity of x+y mod 2.
    U0 = []  # u = x+y for even parity group
    V0 = []  # v = x-y for even parity group
    U1 = []  # u = x+y for odd parity group
    V1 = []  # v = x-y for odd parity group

    for _ in range(n):
        x_str = data.readline().split()
        if not x_str:
            # in case of trailing blank line
            x_str = data.readline().split()
        x = int(x_str[0]); y = int(x_str[1])
        u = x + y
        v = x - y
        if (u & 1) == 0:
            U0.append(u)
            V0.append(v)
        else:
            U1.append(u)
            V1.append(v)

    def sum_pairwise_abs_diff(arr):
        """
        Given a list arr, return sum_{i<j} |arr[j] - arr[i]|.
        """
        arr.sort()
        total = 0
        prefix = 0
        for i, val in enumerate(arr):
            # contribution of val as the larger element in pairs with all previous
            total += val * i - prefix
            prefix += val
        return total

    ans = 0
    # Process each parity group
    for U, V in ((U0, V0), (U1, V1)):
        m = len(U)
        if m < 2:
            continue
        sumU = sum_pairwise_abs_diff(U)
        sumV = sum_pairwise_abs_diff(V)
        # Each Chebyshev distance is (|u_i-u_j| + |v_i-v_j|) / 2.
        ans += (sumU + sumV) // 2

    print(ans)

if __name__ == "__main__":
    main()