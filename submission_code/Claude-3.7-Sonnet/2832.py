class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        value_positions = defaultdict(list)
        for i, num in enumerate(nums):
            value_positions[num].append(i)
        
        max_length = 0
        
        for positions in value_positions.values():
            left = 0
            for right in range(len(positions)):
                # Calculate number of elements to delete using the formula:
                # (positions[right] - positions[left] + 1) - (right - left + 1)
                # which simplifies to (positions[right] - right) - (positions[left] - left)
                while (positions[right] - right) - (positions[left] - left) > k:
                    left += 1
                max_length = max(max_length, right - left + 1)
        
        return max_length