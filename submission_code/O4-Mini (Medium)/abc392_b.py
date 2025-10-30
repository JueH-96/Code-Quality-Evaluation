def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = set(int(next(it)) for _ in range(M))
    missing = [str(i) for i in range(1, N+1) if i not in A]
    C = len(missing)
    # Output
    print(C)
    # If there are missing numbers, print them; otherwise print an empty line
    if C > 0:
        print(" ".join(missing))
    else:
        print()

if __name__ == "__main__":
    main()