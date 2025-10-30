def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    operations = 0
    while sum(a > 0 for a in A) >= 2:
        A.sort(reverse=True)
        A[0] = max(0, A[0] - 1)
        A[1] = max(0, A[1] - 1)
        operations += 1

    print(operations)

if __name__ == "__main__":
    main()