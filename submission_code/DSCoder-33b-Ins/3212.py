class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        prime_counts = [0]*100
        non_prime_counts = [0]*100
        for num in nums:
            if num in primes:
                prime_counts[num] += 1
            else:
                non_prime_counts[num] += 1
        dp = [0]*(len(nums)+1)
        dp[0] = 1
        for i in range(1, len(nums)+1):
            for prime in primes:
                if prime > i:
                    break
                dp[i] += dp[i-prime] * prime_counts[prime]
                dp[i] %= MOD
            for non_prime in range(100):
                if non_prime > i:
                    break
                dp[i] += dp[i-non_prime] * non_prime_counts[non_prime]
                dp[i] %= MOD
        return dp[-1]