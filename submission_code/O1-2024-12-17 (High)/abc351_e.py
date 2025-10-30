def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # We'll separate points into two groups:
    # S0 = points whose (x + y) is even
    # S1 = points whose (x + y) is odd
    # Because a jump cannot change the parity of (x + y),
    # distances between points in different sets are always 0.

    S0 = []
    S1 = []

    idx = 1
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx + 1])
        idx += 2
        if ((x + y) & 1) == 0:
            S0.append((x + y, x - y))
        else:
            S1.append((x + y, x - y))

    # Helper function: given a sorted array L, return sum of |L[j] - L[i]| for all i < j.
    def sum_of_abs_diff(L):
        L.sort()
        prefix_sum = 0
        result = 0
        for i, val in enumerate(L):
            # Add the contribution of val when acting as the "j" in (j - i)
            # i.e. L[j] * (number of L[i] to left) - sum of those L[i].
            result += val * i - prefix_sum
            prefix_sum += val
        return result

    # For points all in the same parity set, the distance between two points (x1,y1) and (x2,y2)
    # is max(|x2 - x1|, |y2 - y1|).  Equivalently, in terms of A=(x+y), B=(x-y),
    #      max(|dx|, |dy|) = (|dA| + |dB|) / 2
    # So sum of pairwise distances in one set S can be computed as
    #      1/2 * [ sum of pairwise |A_i - A_j| + sum of pairwise |B_i - B_j| ].

    def sum_pairwise_linf(points):
        if len(points) < 2:
            return 0
        A = [p[0] for p in points]
        B = [p[1] for p in points]
        sumA = sum_of_abs_diff(A)
        sumB = sum_of_abs_diff(B)
        return (sumA + sumB) // 2

    # Compute the sums for each set and add them.
    ans = sum_pairwise_linf(S0) + sum_pairwise_linf(S1)
    print(ans)

# Do NOT forget to call main()!
main()