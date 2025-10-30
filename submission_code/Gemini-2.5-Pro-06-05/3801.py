class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        
        def count_upto(s_n: str) -> int:
            """
            Counts the number of beautiful positive integers up to the number represented by s_n.
            """
            if s_n == "0":
                return 0
            
            n = len(s_n)
            
            memo = {}

            def dp(index: int, current_sum: int, current_prod: int, is_less: bool, is_started: bool, target_sum: int) -> int:
                """
                The digit DP function.
                State: (index, current_sum, current_prod, is_less, is_started)
                """
                # Pruning: if current sum exceeds target, no valid number can be formed.
                if current_sum > target_sum:
                    return 0
                
                # Base case: we have processed all digits.
                if index == n:
                    # A valid beautiful number must be positive (is_started), have the target sum,
                    # and its product must be divisible by the sum.
                    return 1 if is_started and current_sum == target_sum and current_prod == 0 else 0
                
                # Memoization
                state = (index, current_sum, current_prod, is_less, is_started)
                if state in memo:
                    return memo[state]
                
                res = 0
                # The upper limit for the current digit.
                limit = int(s_n[index]) if not is_less else 9
                
                for digit in range(limit + 1):
                    new_is_less = is_less or (digit < limit)
                    
                    if not is_started:
                        if digit == 0:
                            # Still in the 'not started' state. This path explores numbers with fewer digits.
                            # The product of an empty set is conceptually 1.
                            res += dp(index + 1, 0, 1, new_is_less, False, target_sum)
                        else:
                            # Start a new number with this first non-zero digit.
                            # Sum is the digit itself, product is `digit % target_sum`.
                            res += dp(index + 1, digit, digit % target_sum, new_is_less, True, target_sum)
                    else:  # is_started is True
                        new_sum = current_sum + digit
                        # `current_prod` is already modulo target_sum.
                        new_prod = (current_prod * digit) % target_sum
                        res += dp(index + 1, new_sum, new_prod, new_is_less, True, target_sum)
                
                memo[state] = res
                return res

            total_count = 0
            # Max sum for a number < 10^9 is for 999,999,999, which is 81.
            max_possible_sum = 9 * n
            
            for target_sum in range(1, max_possible_sum + 1):
                memo.clear()  # Reset memoization for each new target_sum
                total_count += dp(0, 0, 1, False, False, target_sum)
            
            return total_count

        return count_upto(str(r)) - count_upto(str(l - 1))