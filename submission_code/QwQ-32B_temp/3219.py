class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a list of tuples (number, original index) and sort by the number
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        
        groups = []
        if not sorted_nums:
            return []
        
        current_group = [sorted_nums[0]]
        for i in range(1, len(sorted_nums)):
            prev_num = sorted_nums[i-1][0]
            curr_num = sorted_nums[i][0]
            if curr_num - prev_num <= limit:
                current_group.append(sorted_nums[i])
            else:
                groups.append(current_group)
                current_group = [sorted_nums[i]]
        # Add the last group
        groups.append(current_group)
        
        res = [0] * len(nums)
        for group in groups:
            # Extract elements and their original indices
            elements = [x[0] for x in group]
            indices = [x[1] for x in group]
            # Sort the indices
            indices.sort()
            # Assign elements to the sorted indices
            for i in range(len(elements)):
                res[indices[i]] = elements[i]
        
        return res