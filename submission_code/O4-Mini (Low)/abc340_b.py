def main():
    import sys
    input = sys.stdin.readline

    Q = int(input().strip())
    A = []
    for _ in range(Q):
        parts = input().split()
        typ = parts[0]
        val = int(parts[1])
        if typ == '1':
            # Append x to the end of A
            A.append(val)
        else:
            # Query: print the k-th value from the end
            k = val
            print(A[-k])

if __name__ == "__main__":
    main()