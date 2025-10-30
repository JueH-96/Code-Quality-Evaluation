import sys

def main() -> None:
    data = list(map(int, sys.stdin.read().split()))
    n = data[0]               # number of elements (not actually needed afterwards)
    a = data[1:]

    total = sum(a)            # total number of “units”
    mx    = max(a)            # largest pile
    rest  = total - mx        # units outside the largest pile

    # We can perform at most total//2 operations because each operation uses 2 units,
    # and at most `rest` operations because each operation must consume one unit
    # from outside the largest pile.
    answer = min(total // 2, rest)

    print(answer)

if __name__ == "__main__":
    main()