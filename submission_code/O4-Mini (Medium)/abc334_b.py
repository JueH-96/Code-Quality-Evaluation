def main():
    import sys
    data = sys.stdin.read().split()
    A = int(data[0])
    M = int(data[1])
    L = int(data[2])
    R = int(data[3])

    # We want integers k such that A + k*M is in [L, R].
    # Solve for k:
    #   L <= A + k*M <= R
    # => (L - A)/M <= k <= (R - A)/M
    # k_min = ceil((L - A)/M)
    # k_max = floor((R - A)/M)

    # floor_div works as Python's // for positive M
    def floor_div(a, b):
        return a // b

    # ceil_div for positive b
    def ceil_div(a, b):
        return -(-a // b)

    k_min = ceil_div(L - A, M)
    k_max = floor_div(R - A, M)

    if k_min > k_max:
        print(0)
    else:
        print(k_max - k_min + 1)

if __name__ == "__main__":
    main()