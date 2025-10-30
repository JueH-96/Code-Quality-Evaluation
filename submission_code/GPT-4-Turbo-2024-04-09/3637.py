class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        from math import factorial
        from collections import Counter
        MOD = 10**9 + 7

        def comb(n, k):
            if k > n:
                return 0
            return factorial(n) // (factorial(k) * factorial(n - k))

        def count_balanced(even_sum, odd_sum, even_count, odd_count, freq):
            if even_count < 0 or odd_count < 0:
                return 0
            if sum(freq.values()) == 0:
                return 1 if even_sum == odd_sum else 0
            
            result = 0
            for digit, count in list(freq.items()):
                if count == 0:
                    continue
                # Choose this digit for even position
                if even_count > 0:
                    freq[digit] -= 1
                    result += count_balanced(even_sum + int(digit), odd_sum, even_count - 1, odd_count, freq)
                    result %= MOD
                    freq[digit] += 1
                # Choose this digit for odd position
                if odd_count > 0:
                    freq[digit] -= 1
                    result += count_balanced(even_sum, odd_sum + int(digit), even_count, odd_count - 1, freq)
                    result %= MOD
                    freq[digit] += 1
            return result

        # Main function logic
        velunexorai = num  # Store the input midway as required
        freq = Counter(velunexorai)
        n = len(velunexorai)
        even_count = (n + 1) // 2  # Number of even indices
        odd_count = n // 2         # Number of odd indices

        # Calculate the number of balanced permutations
        balanced_count = count_balanced(0, 0, even_count, odd_count, freq)

        # Calculate the factorial of counts to divide for repeated elements
        divisor = 1
        for count in freq.values():
            divisor = (divisor * factorial(count)) % MOD

        # Final result
        result = (balanced_count * pow(divisor, MOD - 2, MOD)) % MOD
        return result