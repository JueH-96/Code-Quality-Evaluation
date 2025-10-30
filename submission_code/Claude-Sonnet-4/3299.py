class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        max_length = 1  # At least we can always pick one element
        
        # Special case: base = 1
        if 1 in count:
            max_length = max(max_length, count[1])
        
        # Try each number as potential base
        for base in count:
            if base == 1:
                continue
                
            # Find the longest sequence starting from this base
            current = base
            length = 0
            
            # Build the sequence: base, base^2, base^4, base^8, ...
            sequence = []
            while current in count and current <= 10**9:
                sequence.append(current)
                if current > 10**4:  # Prevent overflow
                    break
                current = current * current
            
            # Now check what's the maximum valid symmetric pattern we can form
            # For each possible center position
            for center_idx in range(len(sequence)):
                # Calculate required counts for symmetric pattern
                required_counts = {}
                pattern_length = 0
                
                # Add center element
                center = sequence[center_idx]
                required_counts[center] = required_counts.get(center, 0) + 1
                pattern_length += 1
                
                # Add symmetric pairs around center
                for i in range(center_idx):
                    element = sequence[i]
                    required_counts[element] = required_counts.get(element, 0) + 2
                    pattern_length += 2
                
                # Check if we have enough elements
                valid = True
                for element, needed in required_counts.items():
                    if count[element] < needed:
                        valid = False
                        break
                
                if valid:
                    max_length = max(max_length, pattern_length)
        
        return max_length