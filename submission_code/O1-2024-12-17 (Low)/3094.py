class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        freq = Counter(nums)
        
        total_ops = 0
        
        for val, count in freq.items():
            # If count < 2, it's impossible to remove these entirely
            if count < 2:
                return -1
            
            # Check remainder of division by 3
            remainder = count % 3
            
            if remainder == 0:
                # All in triples
                total_ops += count // 3
            elif remainder == 1:
                # Use as many triples as possible, then we need an extra 4 to make 2 pairs
                # but we need at least 4 total
                if count < 4:
                    return -1
                total_ops += (count - 4) // 3 + 2
            else:
                # remainder == 2
                # Use as many triples as possible, then one pair
                total_ops += count // 3 + 1
        
        return total_ops