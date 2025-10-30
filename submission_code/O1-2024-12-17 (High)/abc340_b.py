def main():
    import sys

    Q = int(sys.stdin.readline())
    A = []
    for _ in range(Q):
        query = sys.stdin.readline().strip().split()
        typ = int(query[0])
        if typ == 1:
            x = int(query[1])
            A.append(x)
        else:
            k = int(query[1])
            print(A[-k])

# Don't forget to call main()!
if __name__ == "__main__":
    main()