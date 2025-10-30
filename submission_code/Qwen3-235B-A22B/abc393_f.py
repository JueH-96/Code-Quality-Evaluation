import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    q = int(data[ptr])
    ptr += 1

    A = list(map(int, data[ptr:ptr + n]))
    ptr += n

    queries = []
    for i in range(q):
        R_i = int(data[ptr])
        ptr += 1
        X_i = int(data[ptr])
        ptr += 1
        queries.append((R_i, X_i, i))

    queries.sort()

    ans = [0] * q
    tails = []
    j = 0

    for R_i, X_i, idx in queries:
        while j < R_i:
            a = A[j]
            pos = bisect.bisect_left(tails, a)
            if pos < len(tails):
                tails[pos] = a
            else:
                tails.append(a)
            j += 1
        pos = bisect.bisect_right(tails, X_i)
        ans[idx] = pos

    for a in ans:
        print(a)

if __name__ == '__main__':
    main()