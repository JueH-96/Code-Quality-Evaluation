import math

def main():
    N = int(input().strip())
    max_exp = 60
    
    mu = [1] * (max_exp + 1)
    is_prime = [True] * (max_exp + 1)
    primes = []
    for i in range(2, max_exp + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > max_exp:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    total = 0
    for exp in range(2, max_exp + 1):
        if exp == 2:
            r = math.isqrt(N)
        else:
            approx = N ** (1.0 / exp)
            low_cand = max(1, int(approx) - 2)
            high_cand = int(approx) + 2
            r = 1
            for cand in range(high_cand, low_cand - 1, -1):
                try:
                    power = cand ** exp
                except OverflowError:
                    continue
                if power <= N:
                    r = cand
                    break
        count = r - 1
        total += -mu[exp] * count
        
    total += 1
    print(total)

if __name__ == '__main__':
    main()