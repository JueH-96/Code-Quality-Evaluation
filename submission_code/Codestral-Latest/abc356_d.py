MOD = 998244353

def popcount(x):
    return bin(x).count('1')

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    result = 0
    for k in range(N + 1):
        result = (result + popcount(k & M)) % MOD

    print(result)

if __name__ == "__main__":
    main()