class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        import math

        n = len(s)
        counts = Counter(s)

        def get_divisors(n):
            divisors = set()
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n // i)
            return sorted(divisors, reverse=True)

        divisors = get_divisors(n)

        for k in divisors:
            if all(count % k == 0 for count in counts.values()):
                return n // k
        return n