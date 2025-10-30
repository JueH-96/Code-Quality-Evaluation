class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(n, k):
            s = str(n)
            even_count = 0
            odd_count = 0
            for digit in s:
                if int(digit) % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
            return even_count == odd_count and n % k == 0

        def count_beautiful(n, k):
            s = str(n)
            l = len(s)
            dp = {}

            def solve(index, even_count, odd_count, rem, tight):
                if index == l:
                    return 1 if even_count == odd_count and rem == 0 else 0

                if (index, even_count, odd_count, rem, tight) in dp:
                    return dp[(index, even_count, odd_count, rem, tight)]

                ans = 0
                limit = int(s[index]) if tight else 9
                for digit in range(limit + 1):
                    new_even_count = even_count
                    new_odd_count = odd_count
                    if index == 0 and digit == 0:
                        pass
                    elif digit % 2 == 0:
                        new_even_count += 1
                    else:
                        new_odd_count += 1
                    
                    new_rem = (rem * 10 + digit) % k
                    new_tight = tight and (digit == limit)
                    ans += solve(index + 1, new_even_count, new_odd_count, new_rem, new_tight)
                
                dp[(index, even_count, odd_count, rem, tight)] = ans
                return ans

            return solve(0, 0, 0, 0, True)

        return count_beautiful(high, k) - count_beautiful(low - 1, k)