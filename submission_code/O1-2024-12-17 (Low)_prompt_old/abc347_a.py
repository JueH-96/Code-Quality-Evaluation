def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Extract multiples of K and divide them
    result = [x // K for x in A if x % K == 0]
    # Print the quotients in ascending order (they are already in ascending order as A is sorted)
    print(" ".join(map(str, result)))

def main():
    solve()

if __name__ == "__main__":
    main()