class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        from collections import deque
        
        def bit_count(x):
            return bin(x).count("1")
        
        # Create a mapping: bit count -> deque of numbers in their original order.
        groups = {}
        for num in nums:
            k = bit_count(num)
            if k not in groups:
                groups[k] = deque()
            groups[k].append(num)
        
        sorted_nums = sorted(nums)
        # For each number in the sorted order, check if it can be taken from its group.
        for num in sorted_nums:
            k = bit_count(num)
            # If the group is empty or the first element in that group is not the sorted candidate,
            # then reordering within that group is impossible.
            if not groups[k] or groups[k][0] != num:
                return False
            groups[k].popleft()
        return True