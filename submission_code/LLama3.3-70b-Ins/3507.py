class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def count_divisors(n):
            """Count the number of divisors of n."""
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    if n // i == i:
                        count += 1
                    else:
                        count += 2
            return count

        count = 0
        for num in range(l, r + 1):
            # A number is special if it has exactly 3 divisors (1, itself, and one more)
            # This is only true for squares of prime numbers
            # So, we can check if a number is a square and if its square root is prime
            sqrt = int(num**0.5)
            if sqrt * sqrt == num and count_divisors(sqrt) == 2:
                continue
            count += 1
        return count