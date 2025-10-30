class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_num = int(s)
        L = len(s)
        total = 0
        
        # Handle k=0 case
        if start <= s_num <= finish:
            total += 1
        
        k = 1
        while True:
            min_prefix = 10 ** (k - 1)
            max_prefix = limit * (10 ** k - 1) // 9
            min_N = min_prefix * (10 ** L) + s_num
            max_N = max_prefix * (10 ** L) + s_num
            
            if min_N > finish:
                break  # No more possible k
            
            a = max(start, min_N)
            b = min(finish, max_N)
            
            if a > b:
                k += 1
                continue
            
            step = 10 ** L
            # Calculate first_term
            numerator = a - min_N + step - 1
            first_term = min_N + (numerator // step) * step
            # Ensure first_term is within [a, b]
            if first_term < a:
                first_term += step
            
            # Calculate last_term
            last_term = min_N + ((b - min_N) // step) * step
            
            if first_term > last_term:
                k += 1
                continue
            
            count = (last_term - first_term) // step + 1
            total += count
            
            k += 1
        
        return total