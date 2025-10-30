def main():
    import sys
    data = sys.stdin.read().split()
    A = int(data[0])
    M = int(data[1])
    L = int(data[2])
    R = int(data[3])
    # We want integers k such that L <= A + k*M <= R
    # <=> (L - A)/M <= k <= (R - A)/M
    # Since M > 0, we can compute:
    #   k_min = ceil((L - A)/M)
    #   k_max = floor((R - A)/M)
    # Then answer is max(0, k_max - k_min + 1).
    def ceil_div(a, b):
        # b > 0
        return (a + b - 1) // b
    def floor_div(a, b):
        # b > 0
        return a // b

    lo = ceil_div(L - A, M)
    hi = floor_div(R - A, M)
    if hi < lo:
        print(0)
    else:
        print(hi - lo + 1)

if __name__ == "__main__":
    main()