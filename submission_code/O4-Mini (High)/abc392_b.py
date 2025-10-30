def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = set(int(next(it)) for _ in range(M))
    missing = [str(i) for i in range(1, N+1) if i not in A]
    # Output the count
    print(len(missing))
    # Output the missing numbers or a blank line if none
    if missing:
        print(" ".join(missing))
    else:
        print()

main()