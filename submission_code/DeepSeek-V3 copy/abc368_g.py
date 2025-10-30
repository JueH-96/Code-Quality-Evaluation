import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        if data[idx] == '1':
            queries.append((1, int(data[idx+1])-1, int(data[idx+2])))
            idx += 3
        elif data[idx] == '2':
            queries.append((2, int(data[idx+1])-1, int(data[idx+2])))
            idx += 3
        else:
            queries.append((3, int(data[idx+1])-1, int(data[idx+2])-1))
            idx += 3
    for query in queries:
        if query[0] == 1:
            i, x = query[1], query[2]
            A[i] = x
        elif query[0] == 2:
            i, x = query[1], query[2]
            B[i] = x
        else:
            l, r = query[1], query[2]
            v = 0
            for i in range(l, r+1):
                option1 = v + A[i]
                option2 = v * B[i]
                v = max(option1, option2)
            print(v)

if __name__ == "__main__":
    main()