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
    
    count1 = 0
    for p in primes:
        if p**8 <= n:
            count1 += 1
        else:
            break
            
    count2 = 0
    sqrt_n = int(n**0.5)
    for i in range(len(primes)):
        p1 = primes[i]
        if p1 > sqrt_n:
            break
        limit_p2 = sqrt_n // p1
        
        # Count primes p2 such that p1 < p2 <= limit_p2
        start_index = -1
        end_index = -1
        
        # Find index of first prime > p1
        low = i + 1
        high = len(primes) - 1
        first_prime_greater_index = -1
        while low <= high:
            mid = (low + high) // 2
            if primes[mid] > p1:
                first_prime_greater_index = mid
                high = mid - 1
            else:
                low = mid + 1
        
        if first_prime_greater_index == -1:
            continue
            
        start_index = first_prime_greater_index
        
        # Find index of last prime <= limit_p2
        low = start_index
        high = len(primes) - 1
        last_prime_le_limit_index = -1
        while low <= high:
            mid = (low + high) // 2
            if primes[mid] <= limit_p2:
                last_prime_le_limit_index = mid
                low = mid + 1
            else:
                high = mid - 1
                
        if last_prime_le_limit_index == -1:
            continue
            
        end_index = last_prime_le_limit_index
        
        if start_index != -1 and end_index != -1 and start_index <= end_index:
            count2 += (end_index - start_index + 1)
            
    print(count1 + count2)

if __name__ == '__main__':
    solve()