class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element x
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        m = len(nums)
        x = None
        for num, count in freq.items():
            if count * 2 > m:
                x = num
                break
        
        # Compute prefix sums of x's occurrences
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if nums[i] == x else 0)
        total_x = prefix[-1]
        
        # Check each possible split
        for i in range(n - 1):
            left_count = prefix[i+1]
            left_len = i + 1
            if left_count * 2 <= left_len:
                continue
            right_count = total_x - left_count
            right_len = n - i - 1
            if right_count * 2 > right_len:
                return i
        return -1