def main():
    import sys
    input = sys.stdin.readline
    Q = int(input().strip())
    A = []
    output = []
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            # Append operation
            x = int(query[1])
            A.append(x)
        elif query[0] == '2':
            # k-th value from the end
            k = int(query[1])
            output.append(str(A[-k]))
    sys.stdout.write("
".join(output))

if __name__ == "__main__":
    main()