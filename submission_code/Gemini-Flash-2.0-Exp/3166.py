class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        freqs = list(counts.values())
        
        def check(group_size):
            groups = 0
            for freq in freqs:
                num_groups = freq // (group_size + 1)
                rem = freq % (group_size + 1)
                
                if rem > 0 and rem < group_size:
                    return float('inf')
                elif rem > 0:
                    num_groups += 1
                groups += num_groups
            return groups
        
        min_groups = float('inf')
        for group_size in range(1, min(freqs) + 1):
            min_groups = min(min_groups, check(group_size))
        
        return min_groups