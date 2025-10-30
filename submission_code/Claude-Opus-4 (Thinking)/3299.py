class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each number
        count = Counter(nums)
        
        max_length = 0
        
        # Special case for x = 1
        # Since 1^n = 1 for all n, pattern is [1, 1, 1, ...] with odd length
        if 1 in count:
            max_length = count[1] if count[1] % 2 == 1 else count[1] - 1
        
        # Try each number > 1 as the base
        for x in count:
            if x == 1:
                continue
            
            # Build the sequence x, x^2, x^4, x^8, ...
            powers = []
            current = x
            while current in count:
                powers.append(current)
                if current > 31622:  # sqrt(10^9) ≈ 31622 to avoid overflow
                    break
                current = current * current
            
            # Find the longest valid pattern starting with base x
            for i in range(len(powers), 0, -1):
                if i == 1:
                    # Pattern is just [x]
                    length = 1
                else:
                    # Pattern needs 2 of each power except the last one
                    valid = True
                    for j in range(i - 1):
                        if count[powers[j]] < 2:
                            valid = False
                            break
                    if valid and count[powers[i-1]] >= 1:
                        length = 2 * (i - 1) + 1
                    else:
                        continue
                
                max_length = max(max_length, length)
                break
        
        return max_length