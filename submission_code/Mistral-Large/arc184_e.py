import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    A = []
    for _ in range(N):
        row = list(map(int, data[index:index + M]))
        index += M
        A.append(row)

    MOD = 998244353

    def count_flips(a, b):
        flips = 0
        for k in range(M):
            if a[k] != b[k]:
                flips += 1
        return flips

    total_sum = 0
    for i in range(N):
        for j in range(i, N):
            if i == j:
                total_sum += 0
            else:
                flips = count_flips(A[i], A[j])
                if flips % 2 == 0:
                    total_sum += flips // 2
                else:
                    total_sum += (flips // 2) + 1

    print(total_sum % MOD)

solve()