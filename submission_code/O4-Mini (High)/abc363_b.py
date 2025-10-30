def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    T = int(next(it))
    P = int(next(it))
    lengths = [int(next(it)) for _ in range(N)]
    # For each person, compute days needed to reach length T
    needed = [max(0, T - L) for L in lengths]
    needed.sort()
    # The P-th smallest needed days is the answer
    print(needed[P-1])

if __name__ == "__main__":
    main()