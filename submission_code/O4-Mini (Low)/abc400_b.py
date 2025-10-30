import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    N = int(data[0])
    M = int(data[1])
    LIMIT = 10**9

    total = 0
    term = 1  # N^0

    for i in range(M + 1):
        total += term
        if total > LIMIT:
            print("inf")
            return
        # prepare next power
        term *= N
        # Optional early break if term alone already exceeds limit
        # but not strictly necessary since summation check will catch it.
    print(total)


if __name__ == "__main__":
    main()