def main():
    import sys
    data = sys.stdin.read().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    # Take the last K cards and move them to the front
    B = A[-K:] + A[:-K]
    print(*B)

if __name__ == "__main__":
    main()