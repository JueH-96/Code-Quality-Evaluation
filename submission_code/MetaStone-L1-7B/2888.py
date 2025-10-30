class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return -1
        
        # Find the dominant element
        freq = {}
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        d = None
        for key in freq:
            if 2 * freq[key] > n:
                d = key
                break
        
        # Compute prefix sums for 'd'
        prefix = [0] * (n + 1)
        for i in range(n):
            if nums[i] == d:
                prefix[i+1] = prefix[i] + 1
            else:
                prefix[i+1] = prefix[i]
        
        total_d = freq[d]
        # Iterate through possible split points
        for i in range(n-1):
            left_len = i + 1
            left_count = prefix[i+1]
            if 2 * left_count > left_len:
                right_len = n - (i + 1)
                right_count = total_d - left_count
                if 2 * right_count > right_len:
                    return i
        
        return -1