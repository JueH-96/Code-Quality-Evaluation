def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, K = int(data[0]), int(data[1])
    A = list(map(int, data[2:2 + N]))
    # Take the last K elements from A (from the bottom) and move them to the top
    result = A[-K:] + A[:-K]
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()