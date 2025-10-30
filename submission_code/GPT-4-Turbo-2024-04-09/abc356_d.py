def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    MOD = 998244353
    result = 0
    
    for k in range(N + 1):
        result = (result + bin(k & M).count('1')) % MOD
    
    print(result)

if __name__ == "__main__":
    main()