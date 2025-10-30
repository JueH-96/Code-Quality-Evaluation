def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    setA = set(A)
    C = sorted(A + B)
    for i in range(len(C) - 1):
        if C[i] in setA and C[i+1] in setA:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()