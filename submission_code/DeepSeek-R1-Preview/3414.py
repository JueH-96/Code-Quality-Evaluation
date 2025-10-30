class Solution:
    def waysToReachStair(self, k: int) -> int:
        count = 0
        
        # Handle m=0 separately
        if k == 0:
            count += 1
        if k == 1:
            count += 1
        
        # Iterate m from 1 upwards until 2^m exceeds k + 2^m (some upper limit)
        max_m = 0
        if k > 0:
            max_m = k.bit_length() + 2  # Adding 2 to cover edge cases
        
        for m in range(1, max_m + 1):
            s = 1 << m  # 2^m
            denominator = s - 1
            
            # Check equation 1: k = s + n * (s-1)
            if k >= s:
                numerator = k - s
                if denominator == 0:
                    if numerator == 0:
                        count += 1
                else:
                    if numerator % denominator == 0:
                        n = numerator // denominator
                        if n >= 0:
                            count += 1
            
            # Check equation 2: k = s + n * (s-1) -1
            if k + 1 >= s:
                numerator = (k + 1) - s
                if denominator == 0:
                    if numerator == 0:
                        count += 1
                else:
                    if numerator % denominator == 0:
                        n = numerator // denominator
                        if n >= 0:
                            count += 1
        
        return count