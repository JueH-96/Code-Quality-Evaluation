import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + N]))
    index += N
    B = list(map(int, data[index:index + N]))
    index += N
    K = int(data[index])
    index += 1
    XY = []

    for _ in range(K):
        XY.append((int(data[index]), int(data[index + 1])))
        index += 2

    prefix_A = [0] * (N + 1)
    prefix_B = [0] * (N + 1)

    for i in range(1, N + 1):
        prefix_A[i] = prefix_A[i - 1] + A[i - 1]
        prefix_B[i] = prefix_B[i - 1] + B[i - 1]

    results = []

    for X_k, Y_k in XY:
        sum_abs_diff = 0
        for i in range(1, X_k + 1):
            for j in range(1, Y_k + 1):
                sum_abs_diff += abs(A[i - 1] - B[j - 1])
        results.append(sum_abs_diff)

    for result in results:
        print(result)

if __name__ == "__main__":
    solve()