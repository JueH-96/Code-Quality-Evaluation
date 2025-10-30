def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    B = int(data[0])
    # Try all possible A such that A^A could equal B.
    # Since A^A grows very fast, A won't exceed ~60 for B up to 1e18.
    for A in range(1, 61):
        if pow(A, A) == B:
            print(A)
            return
    print(-1)

if __name__ == "__main__":
    main()