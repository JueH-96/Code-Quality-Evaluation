class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(n):
            s = str(n)
            length = len(s)
            
            dp = {}
            
            def solve(index, even_count, odd_count, remainder, is_tight):
                if index == length:
                    if even_count == odd_count and remainder == 0:
                        return 1
                    else:
                        return 0
                
                if (index, even_count, odd_count, remainder, is_tight) in dp:
                    return dp[(index, even_count, odd_count, remainder, is_tight)]
                
                count = 0
                upper_bound = int(s[index]) if is_tight else 9
                
                for digit in range(upper_bound + 1):
                    new_even_count = even_count
                    new_odd_count = odd_count
                    
                    if digit % 2 == 0:
                        new_even_count += 1
                    else:
                        new_odd_count += 1
                    
                    new_remainder = (remainder * 10 + digit) % k
                    new_is_tight = is_tight and (digit == upper_bound)
                    
                    count += solve(index + 1, new_even_count, new_odd_count, new_remainder, new_is_tight)
                
                dp[(index, even_count, odd_count, remainder, is_tight)] = count
                return count
            
            return solve(0, 0, 0, 0, True)
        
        return count_beautiful(high) - count_beautiful(low - 1)