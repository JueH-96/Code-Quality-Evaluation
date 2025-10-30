def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    limit = 10**9

    total = 0
    term = 1
    # sum up N^0 + N^1 + ... + N^M
    for _ in range(M + 1):
        total += term
        if total > limit:
            print("inf")
            return
        term *= N

    print(total)

if __name__ == "__main__":
    main()