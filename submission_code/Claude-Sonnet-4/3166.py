class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each value
        freq = Counter(nums)
        frequencies = list(freq.values())
        
        # Try different group sizes starting from the minimum possible
        min_freq = min(frequencies)
        
        for k in range(min_freq, 0, -1):
            # Try to use groups of size k and k+1
            total_groups = 0
            possible = True
            
            for f in frequencies:
                # For frequency f, we want to split into groups of size k and k+1
                # Let x be number of groups of size k, y be number of groups of size k+1
                # We need: x * k + y * (k+1) = f
                # We want to minimize x + y
                
                # Try all possible values of y (groups of size k+1)
                found = False
                for y in range(f // (k + 1) + 1):
                    remaining = f - y * (k + 1)
                    if remaining >= 0 and remaining % k == 0:
                        x = remaining // k
                        total_groups += x + y
                        found = True
                        break
                
                if not found:
                    possible = False
                    break
            
            if possible:
                return total_groups
        
        # If no valid assignment found, return the sum of all frequencies
        # (each element in its own group)
        return sum(frequencies)