MOD = 998244353

def mod_pow(x, n, mod):
    result = 1
    while n > 0:
        if n % 2 == 1:
            result = result * x % mod
        x = x * x % mod
        n //= 2
    return result

def main():
    K = int(input())
    C = list(map(int, input().split()))

    # Calculate the total number of strings without restrictions
    total = mod_pow(26, K, MOD)

    # Calculate the number of strings that violate the conditions
    # by using the inclusion-exclusion principle
    for i in range(1, 1 << 26):
        count = bin(i).count('1')
        ways = 1
        for j in range(26):
            if i & (1 << j):
                ways *= mod_pow(25, K, MOD)
                ways %= MOD
                ways *= C[j] + 1
                ways %= MOD
        if count % 2 == 1:
            total = (total - ways) % MOD
        else:
            total = (total + ways) % MOD

    print(total)

if __name__ == "__main__":
    main()