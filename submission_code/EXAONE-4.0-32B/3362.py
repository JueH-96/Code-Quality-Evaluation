class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        T = n * (n + 1) // 2
        k_index = (T - 1) // 2
        
        M = 100001
        
        def count_subarrays_at_most_d(d):
            freq = [0] * M
            left = 0
            distinct = 0
            count = 0
            for right in range(n):
                x = nums[right]
                if freq[x] == 0:
                    distinct += 1
                freq[x] += 1
                
                while distinct > d:
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
            if count_subarrays_at_most_d(mid) >= k_index + 1:
                high = mid
            else:
                low = mid + 1
                
        return low