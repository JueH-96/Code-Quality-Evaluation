# YOUR CODE HERE
MOD = 998244353

def mod_inverse(a, mod):
    return pow(a, mod - 2, mod)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    total_time = sum(T)
    
    # Calculate the probability numerator and denominator
    numerator = 0
    for i in range(N):
        if (X + 0.5) % total_time < T[i]:
            numerator += 1
    
    denominator = N
    
    # Calculate the result modulo 998244353
    numerator %= MOD
    denominator %= MOD
    
    result = (numerator * mod_inverse(denominator, MOD)) % MOD
    print(result)

if __name__ == "__main__":
    main()