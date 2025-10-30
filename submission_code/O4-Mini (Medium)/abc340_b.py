def main():
    import sys
    input = sys.stdin.readline

    Q = int(input().strip())
    A = []
    # Process each query
    for _ in range(Q):
        query = input().split()
        t = int(query[0])
        if t == 1:
            # Append x to the end of A
            x = int(query[1])
            A.append(x)
        else:
            # Retrieve the k-th value from the end
            k = int(query[1])
            # Negative indexing: -k gives k-th from end
            print(A[-k])

if __name__ == "__main__":
    main()