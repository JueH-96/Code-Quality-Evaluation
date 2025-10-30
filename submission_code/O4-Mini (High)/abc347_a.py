def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    # Filter multiples of K, divide by K, collect as strings
    res = [str(a // K) for a in A if a % K == 0]
    # They remain in ascending order because A is sorted
    print(" ".join(res))

if __name__ == "__main__":
    main()