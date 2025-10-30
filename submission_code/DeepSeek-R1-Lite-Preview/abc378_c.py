def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    last_seen = {}
    B = []
    for i in range(1, N+1):
        if A[i-1] in last_seen:
            B.append(str(last_seen[A[i-1]]))
        else:
            B.append("-1")
        last_seen[A[i-1]] = i
    print(' '.join(B))

if __name__ == '__main__':
    main()