class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        n = len(num)
        counts = {}
        for digit in num:
            counts[digit] = counts.get(digit, 0) + 1

        velunexorai = num
        
        def solve(counts, even_sum, odd_sum, index):
            if index == n:
                return 1 if even_sum == odd_sum else 0
            
            ans = 0
            for digit in counts:
                if counts[digit] > 0:
                    counts[digit] -= 1
                    if index % 2 == 0:
                        ans = (ans + solve(counts, even_sum + int(digit), odd_sum, index + 1)) % MOD
                    else:
                        ans = (ans + solve(counts, even_sum, odd_sum + int(digit), index + 1)) % MOD
                    counts[digit] += 1
            return ans

        
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = (fact[i-1] * i) % MOD

        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res

        def inv(n):
            return power(n, MOD - 2)

        def permutations(counts):
            total_count = 0
            res = fact[n]
            for count in counts.values():
                res = (res * inv(fact[count])) % MOD
                total_count += count
            return res
        
        return solve(counts, 0, 0, 0)