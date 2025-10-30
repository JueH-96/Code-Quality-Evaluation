def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    
    total_A = 0
    max_diff = -10**18  # A very low number as initial value
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        total_A += a
        # The extra contributed by a giant when placed at the top is (B - A).
        if b - a > max_diff:
            max_diff = b - a

    # The stacking procedure always accumulates all A_i values.
    # The final configuration's top head height equals sum(auto all A_i values) plus an extra from the giant chosen as top.
    # To maximize the height, we pick the giant with the maximum (B - A)
    # Hence, the answer is sum(A_i) + max(B_i - A_i)
    ans = total_A + max_diff
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()