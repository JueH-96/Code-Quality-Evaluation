def solve():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())

    def build_carpet(n):
        if n == 0:
            return ["#"]  # Level-0 is a single black cell
        smaller = build_carpet(n - 1)
        size = 3 ** (n - 1)
        result = []

        # Top row of blocks (3 blocks of smaller carpet)
        for row in smaller:
            result.append(row + row + row)

        # Middle row of blocks (carpet, white block, carpet)
        for row in smaller:
            result.append(row + ("." * size) + row)

        # Bottom row of blocks (3 blocks of smaller carpet)
        for row in smaller:
            result.append(row + row + row)

        return result

    # Build and print the level-N carpet
    carpet = build_carpet(N)
    print("
".join(carpet))

def main():
    solve()

# Call solve() according to the required format
if __name__ == "__main__":
    main()