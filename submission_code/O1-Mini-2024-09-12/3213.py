class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        n = len(nums)
        total = 0

        # Get unique numbers sorted in decreasing order
        unique_nums = sorted(set(nums), reverse=True)
        
        # Track segments where elements are <= current number
        for m in unique_nums:
            positions = [i for i, num in enumerate(nums) if num == m]
            if len(positions) < k:
                continue
            
            start = 0
            while start < n:
                # Skip elements greater than m
                while start < n and nums[start] > m:
                    start += 1
                if start >= n:
                    break
                end = start
                while end < n and nums[end] <= m:
                    end += 1
                
                # Extract positions of m within the current segment
                seg_positions = [p for p in positions if start <= p < end]
                if len(seg_positions) >= k:
                    # Total subarrays in the segment
                    length = end - start
                    total_subarrays = length * (length + 1) // 2
                    
                    # Number of subarrays with at most k-1 m's
                    count = 0
                    left = start
                    cnt = 0
                    for right in range(start, end):
                        if nums[right] == m:
                            cnt += 1
                        while cnt > k - 1:
                            if nums[left] == m:
                                cnt -= 1
                            left += 1
                        count += right - left + 1
                        
                    # Subarrays with at least k m's
                    total += total_subarrays - count
                start = end
        return total