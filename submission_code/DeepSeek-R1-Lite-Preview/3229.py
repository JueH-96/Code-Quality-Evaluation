class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Function to find the closest palindrome to a given number m
        def closest_palindrome(m: int) -> int:
            s = str(m)
            n = len(s)
            half = s[:(n + 1) // 2]
            candidates = set()
            
            # Generate candidates by adjusting the first half by -1, 0, and +1
            for i in range(-1, 2):
                new_half = str(int(half) + i)
                if len(new_half) != len(half):
                    if len(new_half) < len(half):
                        prefix = new_half
                        suffix = prefix[:-1][::-1] if n % 2 else prefix[::-1]
                    else:
                        prefix = new_half
                        suffix = prefix[:-1][::-1] if n % 2 else prefix[::-1]
                else:
                    prefix = new_half
                    suffix = prefix[:-1][::-1] if n % 2 else prefix[::-1]
                candidate = int(prefix + suffix)
                candidates.add(candidate)
            
            # Consider edge cases
            candidates.add(10**(n - 1) - 1)
            candidates.add(10**n + 1)
            
            # Filter out candidates that are >= 10^9
            candidates = [c for c in candidates if c < 10**9]
            
            # Find the closest palindrome to m
            closest = None
            min_diff = float('inf')
            for c in candidates:
                if c == m:
                    return c
                diff = abs(c - m)
                if diff < min_diff or (diff == min_diff and c < closest):
                    min_diff = diff
                    closest = c
            return closest
        
        # Find the median of the array
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        if n % 2 == 1:
            median = nums_sorted[n // 2]
        else:
            median = nums_sorted[n // 2 - 1]  # Choose the lower median
        
        # Find the closest palindrome to the median
        y = closest_palindrome(median)
        
        # Calculate the total cost
        total_cost = sum(abs(num - y) for num in nums)
        
        return total_cost