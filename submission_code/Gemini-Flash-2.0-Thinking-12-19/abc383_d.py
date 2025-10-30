def solve():
    n = int(input())
    limit_primes = 2 * 10**6
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for p in range(2, limit_primes + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, limit_primes + 1, p):
                is_prime[i] = False
    
    count = 0
    
    # Case 1: numbers of the form p^8
    for p in primes:
        val = p**8
        if val <= n:
            count += 1
        else:
            break
            
    # Case 2: numbers of the form p1^2 * p2^2 = (p1*p2)^2, with p1 < p2
    sqrt_n = int(n**0.5)
    n_quart_root = int(n**0.25) + 2 # add 2 for safety margin
    
    primes_p1 = []
    for p in primes:
        if p > n_quart_root:
            break
        primes_p1.append(p)
        
    for i in range(len(primes_p1)):
        p1 = primes_p1[i]
        limit_p2 = sqrt_n // p1
        for j in range(i + 1, len(primes)):
            p2 = primes[j]
            if p2 > limit_p2:
                break
            count += 1
            
    print(count)

if __name__ == '__main__':
    solve()