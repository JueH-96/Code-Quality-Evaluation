class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Helper function to check if a number is prime
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x ** 0.5) + 1):
                if x % i == 0:
                    return False
            return True

        # Initialize the result list
        result = []

        # Iterate through all possible pairs (x, y) where x <= y <= n
        for x in range(1, n // 2 + 1):
            y = n - x
            if is_prime(x) and is_prime(y):
                result.append([x, y])

        # Sort the result list
        result.sort()
        return result