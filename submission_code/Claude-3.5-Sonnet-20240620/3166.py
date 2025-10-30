class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        
        def can_group(size):
            return all((count // size + (1 if count % size else 0)) * (size - 1) <= count for count in counts.values())
        
        def group_count(size):
            return sum((count // size + (1 if count % size else 0)) for count in counts.values())
        
        counts = Counter(nums)
        min_count = min(counts.values())
        
        for size in range(min_count, 0, -1):
            if can_group(size):
                return group_count(size)
        
        return len(nums)