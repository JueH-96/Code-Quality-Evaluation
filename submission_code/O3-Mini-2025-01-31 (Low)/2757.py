class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10**9 + 7
        
        # Helper function: subtract one from numeric string
        def dec_str(num: str) -> str:
            # converts string num to string representing num - 1
            num_list = list(num)
            i = len(num_list) - 1
            while i >= 0:
                if num_list[i] != '0':
                    num_list[i] = str(int(num_list[i]) - 1)
                    break
                else:
                    num_list[i] = '9'
                    i -= 1
            # Remove any leading zeros if exists
            res = "".join(num_list).lstrip("0")
            return res if res != "" else "0"
        
        from functools import lru_cache
        
        def count_up_to(num: str) -> int:
            # digit DP: count numbers with digit sum in [min_sum, max_sum] and <= num
            digits = [int(d) for d in num]
            n = len(digits)
            
            @lru_cache(maxsize=None)
            def dp(pos: int, tight: int, cur_sum: int) -> int:
                if pos == n:
                    # check if the current sum is within bounds, note that 0 is allowed if it falls in [min_sum, max_sum],
                    # but since our input num1 is >= 1, we handle 0 in difference appropriately.
                    return 1 if (min_sum <= cur_sum <= max_sum) else 0
                
                limit = digits[pos] if tight else 9
                res = 0
                for dig in range(0, limit+1):
                    new_tight = tight and (dig == limit)
                    new_sum = cur_sum + dig
                    # As an optimization, if new_sum already > max_sum, then none of the completions will be valid.
                    # Also, maximum additional sum possible from remaining digits is 9*(n-pos-1).
                    if new_sum > max_sum:
                        continue
                    max_possible = new_sum + 9 * (n - pos - 1)
                    if max_possible < min_sum:
                        continue
                    res = (res + dp(pos+1, new_tight, new_sum)) % mod
                return res
            
            return dp(0, 1, 0)
        
        # Count good numbers in [num1, num2]
        # use count_up_to to count numbers <= X with the property.
        count2 = count_up_to(num2)
        # For lower bound: count numbers <= (num1 - 1)
        if num1 == "0" or num1 == "1":
            count1 = 0
        else:
            prev = dec_str(num1)
            count1 = count_up_to(prev)
        
        return (count2 - count1) % mod

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.count("1", "12", 1, 8))  # Expected output: 11
    print(sol.count("1", "5", 1, 5))   # Expected output: 5