# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def main():
    N, X = map(int, input().split())
    T = list(map(int, input().split()))
    
    total_time = sum(T)
    prob_choose = mod_inverse(N)
    
    X += 0.5
    cycles = X // total_time
    X %= total_time
    
    numerator = 0
    denominator = 1
    
    for i in range(N):
        start = sum(T[:i])
        end = start + T[i]
        
        if start <= X < end:
            numerator = T[0]
            denominator = total_time
            break
        
        if i == 0 and X >= end:
            numerator = 0
            denominator = 1
            break
    
    if cycles > 0:
        numerator = (numerator * pow(N, cycles, MOD)) % MOD
        denominator = (denominator * pow(N, cycles, MOD)) % MOD
    
    result = (numerator * mod_inverse(denominator)) % MOD
    print(result)

if __name__ == "__main__":
    main()