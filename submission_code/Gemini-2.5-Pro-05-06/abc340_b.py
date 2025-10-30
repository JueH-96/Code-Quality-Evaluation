def main():
    Q = int(input())
    A = []

    for _ in range(Q):
        query_parts = input().split()
        query_type = int(query_parts[0])

        if query_type == 1:
            # Query type 1: Append x to the end of A
            x = int(query_parts[1])
            A.append(x)
        elif query_type == 2:
            # Query type 2: Find the k-th value from the end of A
            k = int(query_parts[1])
            # Python's negative indexing A[-k] accesses the k-th element from the end.
            # For example, A[-1] is the last element, A[-2] is the second to last, etc.
            # This matches the problem's definition of "k-th from the end".
            # It's guaranteed that the length of A is at least k.
            value = A[-k]
            print(value)

if __name__ == '__main__':
    main()