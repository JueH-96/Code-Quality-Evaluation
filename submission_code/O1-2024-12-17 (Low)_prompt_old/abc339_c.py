def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    prefix_sum = 0
    min_prefix = float('inf')
    total = 0

    for delta in A:
        prefix_sum += delta
        if prefix_sum < min_prefix:
            min_prefix = prefix_sum
    total = prefix_sum  # sum(A)

    # X must ensure all intermediate sums are non-negative
    # i.e. X + prefix_sum(i) >= 0 for all i -> X >= -min_prefix when min_prefix < 0
    # also final number of passengers >= 0 -> X >= -total (if total < 0)
    # thus X >= max(0, -min_prefix, -total)
    X = max(0, -min_prefix, -total)
    # The final number of passengers is total + X
    print(total + X)

def main():
    solve()

if __name__ == "__main__":
    main()