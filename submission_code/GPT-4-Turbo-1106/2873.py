class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        def is_prime(num):
            if num <= 1:
                return False
            if num <= 3:
                return True
            if num % 2 == 0 or num % 3 == 0:
                return False
            i = 5
            while i * i <= num:
                if num % i == 0 or num % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        prime_pairs = []
        for x in range(1, n // 2 + 1):
            y = n - x
            if is_prime(x) and is_prime(y):
                prime_pairs.append([x, y])
        
        return prime_pairs