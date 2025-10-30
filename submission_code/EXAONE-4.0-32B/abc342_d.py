import math
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    MAX_VAL = 200000
    sq_free = list(range(MAX_VAL + 1))
    
    max_p = int(math.isqrt(MAX_VAL))
    primes = []
    if max_p >= 2:
        is_prime = [True] * (max_p + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, max_p + 1):
            if is_prime[i]:
                primes.append(i)
                start = i * i
                if start <= max_p:
                    for j in range(start, max_p + 1, i):
                        is_prime[j] = False
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_VAL:
            break
        j = p2
        while j <= MAX_VAL:
            while sq_free[j] % p2 == 0:
                sq_free[j] //= p2
            j += p2
            
    zeros = 0
    freq = {}
    for num in arr:
        if num == 0:
            zeros += 1
        else:
            s = sq_free[num]
            freq[s] = freq.get(s, 0) + 1
            
    total_pairs = zeros * (zeros - 1) // 2
    total_pairs += zeros * (n - zeros)
    
    for count in freq.values():
        total_pairs += count * (count - 1) // 2
        
    print(total_pairs)

if __name__ == "__main__":
    main()