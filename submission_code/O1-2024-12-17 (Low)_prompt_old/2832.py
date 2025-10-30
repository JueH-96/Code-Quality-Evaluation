class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary to hold positions of each distinct value
        positions = defaultdict(list)
        
        # Populate positions of each value
        for i, val in enumerate(nums):
            positions[val].append(i)
        
        answer = 1  # At least a single element (or empty subarray) can always form an equal subarray
        
        # For each distinct value, use a two-pointer approach to find the longest subarray
        # satisfying: (pos[right] - pos[left] + 1) - (right - left + 1) <= k
        # which is equivalent to (pos[right] - right) - (pos[left] - left) <= k
        for val, pos_list in positions.items():
            left = 0
            for right in range(len(pos_list)):
                # Maintain the condition by moving 'left' if needed
                while pos_list[right] - right - (pos_list[left] - left) > k:
                    left += 1
                # The subarray length is pos_list[right] - pos_list[left] + 1
                current_length = pos_list[right] - pos_list[left] + 1
                answer = max(answer, current_length)
        
        return answer