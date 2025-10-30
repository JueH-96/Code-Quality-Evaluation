class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each value
        freq = Counter(nums)
        counts = list(freq.values())
        
        # If all elements are the same, we need at least 1 group
        if len(counts) == 1:
            # We can have groups of size k and k+1
            # Try different values of k to minimize groups
            n = counts[0]
            min_groups = n  # worst case: each element in its own group
            
            for k in range(1, n + 1):
                # Check if we can divide n into groups of size k and k+1
                # Let x be number of groups of size k+1, y be number of groups of size k
                # x * (k+1) + y * k = n
                # x + y = total groups
                
                # We want to minimize x + y
                # From the equation: x = (n - k*y) / (k+1)
                # For valid solution, (n - k*y) must be divisible by (k+1) and x >= 0
                
                groups = float('inf')
                for y in range(n // k + 1):
                    if (n - k * y) % (k + 1) == 0:
                        x = (n - k * y) // (k + 1)
                        if x >= 0:
                            groups = min(groups, x + y)
                
                if groups != float('inf'):
                    min_groups = min(min_groups, groups)
            
            return min_groups
        
        # For multiple distinct values, we need to find a common group size
        # that works for all frequencies
        min_count = min(counts)
        
        # Try different group sizes from 1 to min_count + 1
        for size in range(min_count, 0, -1):
            # Check if this size works for all counts
            total_groups = 0
            valid = True
            
            for count in counts:
                # Try to divide count into groups of size 'size' and 'size+1'
                # We need to check if it's possible
                min_groups_for_count = float('inf')
                
                for num_small in range(count // size + 1):
                    remaining = count - num_small * size
                    if remaining % (size + 1) == 0:
                        num_large = remaining // (size + 1)
                        min_groups_for_count = min(min_groups_for_count, num_small + num_large)
                
                if min_groups_for_count == float('inf'):
                    valid = False
                    break
                
                total_groups += min_groups_for_count
            
            if valid:
                return total_groups
        
        # This should never happen with valid input
        return len(nums)