def main():
    import sys
    input = sys.stdin.readline
    Q = int(input())
    A = []
    
    for _ in range(Q):
        query = input().strip().split()
        if query[0] == "1":
            # Append x to the sequence A
            A.append(int(query[1]))
        elif query[0] == "2":
            # Print the k-th value from the end of A
            k = int(query[1])
            print(A[-k])

if __name__ == '__main__':
    main()