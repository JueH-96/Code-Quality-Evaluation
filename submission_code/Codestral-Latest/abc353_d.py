# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    total_sum = 0

    for i in range(N):
        for j in range(i + 1, N):
            z = int(str(A[i]) + str(A[j]))
            total_sum = (total_sum + z) % MOD

    print(total_sum)

if __name__ == "__main__":
    main()