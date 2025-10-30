class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n  # Initialize with the maximum possible operations (delete all digits)
        endings = ["00", "25", "50", "75"]
        
        for end in endings:
            a, b = end[0], end[1]
            # Find the rightmost occurrence of the second digit (b)
            b_pos = -1
            for i in range(n-1, -1, -1):
                if num[i] == b:
                    b_pos = i
                    break
            if b_pos == -1:
                continue  # No second digit found
            
            # Find the rightmost occurrence of the first digit (a) before b_pos
            a_pos = -1
            for i in range(b_pos - 1, -1, -1):
                if num[i] == a:
                    a_pos = i
                    break
            if a_pos == -1:
                continue  # No first digit found before b_pos
            
            # Calculate the number of deletions needed
            cost = n - a_pos - 2
            if cost < min_ops:
                min_ops = cost
        
        # Check for the case where we can leave a single zero
        has_zero = '0' in num
        if has_zero:
            min_ops = min(min_ops, n - 1)
        
        return min_ops