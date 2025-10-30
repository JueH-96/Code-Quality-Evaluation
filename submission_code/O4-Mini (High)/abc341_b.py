def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Read initial amounts of each currency
    B = [int(next(it)) for _ in range(N)]
    # For each conversion rule from country i to i+1,
    # convert as much as possible greedily.
    for i in range(N - 1):
        S = int(next(it))
        T = int(next(it))
        # number of times we can apply the conversion at country i
        ops = B[i] // S
        # increase country (i+1)'s currency
        B[i + 1] += ops * T
    # The answer is the final amount in country N
    print(B[N - 1])

if __name__ == "__main__":
    main()