class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        # Helper to subtract one from a numeric string (assuming numStr >= "1").
        def subtract_one(numStr: str) -> str:
            # Special case: if numStr == "0", just return "0"
            # (the problem constraints state num1 >= "1", so normally not needed,
            #  but let's be robust).
            if numStr == "0":
                return "0"
            
            digits = list(numStr)
            i = len(digits) - 1
            # Subtract one from the rightmost digit
            while i >= 0 and digits[i] == '0':
                digits[i] = '9'
                i -= 1
            if i >= 0:
                # Decrement the digit
                digits[i] = str(int(digits[i]) - 1)
            
            # Remove leading zeros if any
            result = "".join(digits).lstrip('0')
            return result if result != "" else "0"
        
        # Digit DP to count how many integers in [0..numStr] have digit-sum in [min_sum, max_sum].
        def count_up_to(numStr: str, minSum: int, maxSum: int) -> int:
            # Edge case: if numStr == "0", we only have x = 0
            # sum of digits is 0, so check if that is in [minSum..maxSum].
            if numStr == "0":
                return 1 if (minSum <= 0 <= maxSum) else 0
            
            digits = [int(c) for c in numStr]
            n = len(digits)
            # dp memo: key = (pos, current_sum, tight), value = count
            from functools import lru_cache
            
            @lru_cache(None)
            def dfs(pos, current_sum, tight):
                if pos == n:
                    # If we've placed all digits, check if sum is in [minSum..maxSum].
                    return 1 if (minSum <= current_sum <= maxSum) else 0
                
                limit = digits[pos] if tight else 9
                ways = 0
                for dig in range(limit + 1):
                    nxt_sum = current_sum + dig
                    # If sum already exceeds maxSum, no need to go further
                    if nxt_sum > maxSum:
                        break
                    ways += dfs(pos + 1, nxt_sum, tight and (dig == limit))
                return ways % MOD
            
            return dfs(0, 0, True)
        
        # Main logic: count good numbers in [num1..num2].
        # = count_in [0..num2] - count_in [0..(num1-1)]
        # Take care if num1 == "1" => num1 - 1 = "0" => handle properly.
        num1_minus_1 = subtract_one(num1)
        
        ans = count_up_to(num2, min_sum, max_sum) - count_up_to(num1_minus_1, min_sum, max_sum)
        ans %= MOD
        return ans