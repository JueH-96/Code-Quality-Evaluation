def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A, M, L, R = map(int, input_data)

    # Define a function for ceiling division when divisor is positive.
    # This returns ceil(a/b) for integer a.
    def ceil_div(a, b):
        return -( -a // b )

    # For a tree at position X = A + k*M to lie between L and R,
    # we must have L <= A + k*M <= R.
    # Solve for k:
    #   k >= ceil((L - A) / M)
    #   k <= floor((R - A) / M)
    k_min = ceil_div(L - A, M)
    k_max = (R - A) // M

    # The count is valid only if there exists an integer k that satisfies the conditions.
    if k_max < k_min:
        print(0)
    else:
        print(k_max - k_min + 1)

if __name__ == '__main__':
    main()