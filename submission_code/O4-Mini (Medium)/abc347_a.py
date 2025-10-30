def main():
    import sys
    data = sys.stdin.read().split()
    # First two values are N and K
    N, K = map(int, data[:2])
    # The rest are the sequence A
    A = map(int, data[2:])
    # Filter multiples of K, divide by K, collect as strings
    res = [str(a // K) for a in A if a % K == 0]
    # Print in ascending order separated by spaces
    print(" ".join(res))

if __name__ == "__main__":
    main()