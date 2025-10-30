MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    
    if N == 1 and M == 1:
        print(1)
    elif N == 1:
        result = (M * M) % MOD
        print(result)
    elif M == 1:
        result = (2 * N - 1) % MOD
        print(result)
    else:
        NM = N * M
        result = NM * (NM + 1) % MOD
        print(result)

if __name__ == '__main__':
    main()