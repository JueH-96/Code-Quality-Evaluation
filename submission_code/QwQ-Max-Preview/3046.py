class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = float('inf')
        
        # Check each possible suffix pair
        for suffix in ['00', '25', '50', '75']:
            a, b = suffix[0], suffix[1]
            # Find the last occurrence of b
            j = n - 1
            while j >= 0:
                if num[j] == b:
                    break
                j -= 1
            if j == -1:
                continue  # No occurrence of b found
            # Find the last occurrence of a before j
            i = j - 1
            while i >= 0:
                if num[i] == a:
                    break
                i -= 1
            if i == -1:
                continue  # No occurrence of a found before j
            # Calculate steps for this pair
            steps = (n - 1 - j) + (j - i - 1)
            if steps < min_ops:
                min_ops = steps
        
        # Check the zero case
        has_zero = any(c == '0' for c in num)
        if has_zero:
            steps_zero = n - 1
        else:
            steps_zero = n
        if steps_zero < min_ops:
            min_ops = steps_zero
        
        return min_ops