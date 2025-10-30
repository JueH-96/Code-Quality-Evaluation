class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        k_index = (total_subarrays - 1) // 2
        
        def at_most(k):
            from collections import defaultdict
            freq = defaultdict(int)
            left = 0
            distinct = 0
            count = 0
            for right in range(n):
                x = nums[right]
                if freq[x] == 0:
                    distinct += 1
                freq[x] += 1
                while distinct > k and left <= right:
                    y = nums[left]
                    freq[y] -= 1
                    if freq[y] == 0:
                        distinct -= 1
                    left += 1
                count += (right - left + 1)
            return count
        
        low, high = 1, n
        while low < high:
            mid = (low + high) // 2
            if at_most(mid) >= k_index + 1:
                high = mid
            else:
                low = mid + 1
        return low