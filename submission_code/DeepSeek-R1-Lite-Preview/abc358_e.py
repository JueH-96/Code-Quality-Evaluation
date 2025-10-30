MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    K = int(data[0])
    Ci = list(map(int, data[1:27]))
    
    gf = [0] * (K + 1)
    gf[0] = 1
    
    for c in Ci:
        # Compute prefix sums of gf
        prefix = [0] * (K + 2)
        prefix[0] = gf[0]
        for i in range(1, K + 1):
            prefix[i] = (prefix[i - 1] + gf[i]) % MOD
        # Update gf using the prefix sums
        for l in range(K + 1):
            if l - c - 1 >= 0:
                gf[l] = (prefix[l] - prefix[l - c - 1]) % MOD
            else:
                gf[l] = prefix[l] % MOD
        # Ensure non-negative values
        gf[l] = (gf[l] + MOD) % MOD
    
    # Sum from gf[1] to gf[K] to get the total number of valid strings
    result = sum(gf[1:]) % MOD
    print(result)

if __name__ == '__main__':
    main()