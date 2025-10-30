class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        max_len = 0
        for num in sorted(counts.keys()):
            curr_len = 0
            
            # Try starting with num
            curr = num
            curr_counts = counts.copy()
            while curr in curr_counts and curr_counts[curr] > 0:
                curr_len += 1
                curr_counts[curr] -= 1
                curr += 1
            max_len = max(max_len, curr_len)
            
            # Try starting with num + 1
            curr_len = 0
            curr = num + 1
            curr_counts = counts.copy()
            if num in curr_counts and curr_counts[num] > 0:
                curr_counts[num] -= 1
                curr_len += 1
                while curr in curr_counts and curr_counts[curr] > 0:
                    curr_len += 1
                    curr_counts[curr] -= 1
                    curr += 1
                max_len = max(max_len, curr_len)
            
        return max_len