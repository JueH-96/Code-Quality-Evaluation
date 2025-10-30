import sys

MOD = 998244353

def expected_value(N, K):
    # The expected position of the black ball after K operations
    # is simply the average of all possible positions, which is (N + 1) / 2.
    # Since we need the result modulo 998244353, we compute it as follows:
    expected_pos = (N + 1) * pow(2, MOD - 2, MOD) % MOD
    return expected_pos

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    result = expected_value(N, K)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()