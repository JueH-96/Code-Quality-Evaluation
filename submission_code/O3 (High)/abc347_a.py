def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    quotients = [a // K for a in A if a % K == 0]

    print(" ".join(map(str, quotients)))

if __name__ == "__main__":
    main()