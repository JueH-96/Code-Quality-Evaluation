import functools

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        
        # This helper function counts beautiful integers from 1 up to 'n_str' (inclusive).
        def count_beautiful_up_to(n_str: str) -> int:
            N = len(n_str)
            
            # Memoization for the DP states.
            # Using functools.lru_cache automatically handles the dictionary creation and lookup.
            @functools.lru_cache(None)
            def dp(idx: int, tight: bool, is_leading_zero: bool, even_count: int, odd_count: int, current_sum_mod_k: int) -> int:
                
                # Base case: All digits processed.
                if idx == N:
                    # If is_leading_zero is True, it means we only placed leading zeros,
                    # effectively forming the number 0 (e.g., processing "000" when target is "123").
                    # The problem asks for positive integers, so 0 is not beautiful, nor are "empty" numbers.
                    if is_leading_zero:
                        return 0
                    
                    # A non-zero number has been formed. Check if it's beautiful.
                    if even_count == odd_count and current_sum_mod_k == 0:
                        return 1
                    return 0
                
                res = 0
                # Determine the upper limit for the current digit.
                # If 'tight' is True, we are restricted by the digit at n_str[idx].
                # Otherwise, we can place any digit up to 9.
                upper_bound = int(n_str[idx]) if tight else 9

                for digit in range(upper_bound + 1):
                    # Case 1: Still placing leading zeros.
                    # If we are currently in the leading zero phase AND the current digit is 0,
                    # then this '0' is a leading zero. It does not contribute to even/odd counts
                    # or the number's value modulo k.
                    if is_leading_zero and digit == 0:
                        # Recurse: move to next index, 'tight' remains based on current digit,
                        # 'is_leading_zero' remains True, counts and sum_mod_k remain 0.
                        res += dp(idx + 1, tight and (digit == upper_bound), True, 0, 0, 0)
                    else:
                        # Case 2: Placing the first non-zero digit or subsequent digits.
                        # This digit contributes to the number's properties.
                        new_even_count = even_count + (1 if digit % 2 == 0 else 0)
                        new_odd_count = odd_count + (1 if digit % 2 != 0 else 0)
                        new_sum_mod_k = (current_sum_mod_k * 10 + digit) % k
                        
                        # Recurse: move to next index, 'tight' is updated, 'is_leading_zero' becomes False.
                        res += dp(idx + 1, tight and (digit == upper_bound), False, new_even_count, new_odd_count, new_sum_mod_k)
                
                return res

            # Initial call to the DP function for a given n_str.
            # Start from index 0.
            # 'tight' is True initially, as we are restricted by the input number's digits.
            # 'is_leading_zero' is True initially, as we haven't placed any non-zero digits yet.
            # Counts and sum_mod_k are 0.
            result = dp(0, True, True, 0, 0, 0)
            
            # Clear the cache for this specific `dp` function instance.
            # This is important because `n_str` changes between calls to `count_beautiful_up_to`
            # (e.g., for `high` and `low - 1`), but `n_str` is not part of the memoized state.
            # By defining `dp` inside `count_beautiful_up_to`, each call to `count_beautiful_up_to`
            # creates a new `dp` function and its own cache, so `cache_clear` is technically
            # redundant here for correctness, but it's good practice in case the structure changes.
            dp.cache_clear() 
            return result
        
        # The number of beautiful integers in the range [low, high] is
        # (count of beautiful integers up to high) - (count of beautiful integers up to low - 1).
        return count_beautiful_up_to(str(high)) - count_beautiful_up_to(str(low - 1))