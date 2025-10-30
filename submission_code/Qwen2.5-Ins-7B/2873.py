class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        primes = [num for num in range(2, n) if is_prime(num)]
        result = []
        seen = set()
        
        for prime in primes:
            complement = n - prime
            if complement in primes and (complement, prime) not in seen:
                result.append([prime, complement])
                seen.add((prime, complement))
        
        return result