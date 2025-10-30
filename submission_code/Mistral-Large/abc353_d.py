import sys

MOD = 998244353

def f(x, y):
    return int(str(x) + str(y))

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    total_sum = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            total_sum += f(A[i], A[j])
            total_sum %= MOD

    print(total_sum)

if __name__ == "__main__":
    main()