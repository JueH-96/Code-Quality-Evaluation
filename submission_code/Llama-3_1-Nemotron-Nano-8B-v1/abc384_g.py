import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    K = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(K):
        X = int(input[ptr])
        Y = int(input[ptr+1])
        queries.append((X, Y))
        ptr += 2

    for X, Y in queries:
        if Y == 0 or X == 0:
            print(0)
            continue
        B_sub = B[:Y]
        B_sub.sort()
        prefix_B = [0]
        for b in B_sub:
            prefix_B.append(prefix_B[-1] + b)
        sum_total = 0
        for i in range(X):
            a = A[i]
            cnt_le = bisect.bisect_right(B_sub, a)
            sum_le = prefix_B[cnt_le]
            sum_gt = prefix_B[-1] - sum_le
            contribution = (a * cnt_le - sum_le) + (sum_gt - a * (Y - cnt_le))
            sum_total += contribution
        print(sum_total)

if __name__ == '__main__':
    main()