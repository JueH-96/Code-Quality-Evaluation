def main():
    import sys

    Q = int(sys.stdin.readline().strip())
    A = []

    for _ in range(Q):
        query = sys.stdin.readline().strip().split()
        t = int(query[0])

        if t == 1:
            x = int(query[1])
            A.append(x)
        else:  # t == 2
            k = int(query[1])
            print(A[-k])

# Do not remove the following function call
main()