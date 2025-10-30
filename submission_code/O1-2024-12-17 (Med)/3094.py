class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        freq = Counter(nums)
        
        total_ops = 0
        for val, count in freq.items():
            # If there's only one occurrence of a value, it's impossible to remove
            if count == 1:
                return -1
            
            # Compute how many triples we can form
            triples = count // 3
            remainder = count % 3
            
            if remainder == 0:
                # All occurrences can be removed in triples
                total_ops += triples
            elif remainder == 1:
                # If count < 4, we cannot fix a remainder of 1
                if count < 4:
                    return -1
                # Remove one triple and convert leftover 4 into two pairs
                # total_ops = (triples - 1) + 2 = triples + 1
                total_ops += (triples + 1)
            else:  # remainder = 2
                # We can remove leftover 2 as one pair
                total_ops += (triples + 1)
        
        return total_ops