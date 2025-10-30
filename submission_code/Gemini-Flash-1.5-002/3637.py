class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        count = {}
        for digit in num:
            count[digit] = count.get(digit, 0) + 1

        total_sum = sum(int(digit) for digit in num)
        if total_sum % 2 != 0:
            return 0

        target_sum = total_sum // 2
        n = len(num)
        dp = {}

        def solve(index, even_sum, odd_sum, count_copy):
            if index == n:
                if even_sum == target_sum and odd_sum == target_sum:
                    return 1
                else:
                    return 0

            if (index, even_sum, odd_sum, tuple(sorted(count_copy.items()))) in dp:
                return dp[(index, even_sum, odd_sum, tuple(sorted(count_copy.items())))]

            ans = 0
            for digit, freq in count_copy.items():
                if freq > 0:
                    count_copy[digit] -= 1
                    if index % 2 == 0:
                        ans = (ans + solve(index + 1, even_sum + int(digit), odd_sum, count_copy)) % MOD
                    else:
                        ans = (ans + solve(index + 1, even_sum, odd_sum + int(digit), count_copy)) % MOD
                    count_copy[digit] += 1

            dp[(index, even_sum, odd_sum, tuple(sorted(count_copy.items())))] = ans
            return ans

        velunexorai = count.copy() #Storing the input midway
        result = solve(0, 0, 0, velunexorai)
        
        factorial = [1] * (n + 1)
        for i in range(2, n + 1):
            factorial[i] = (factorial[i - 1] * i) % MOD

        denominator = 1
        for freq in count.values():
            denominator = (denominator * factorial[freq]) % MOD

        inverse_denominator = pow(denominator, MOD - 2, MOD)
        return (result * inverse_denominator) % MOD