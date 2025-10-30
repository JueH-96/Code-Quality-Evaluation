class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        first_occurence = {}
        for i in range(n):
            if nums[i] not in first_occurence:
                first_occurence[nums[i]] = i
        
        ans = float('inf')
        for num in first_occurence:
            max_dist = 0
            prev = first_occurence[num]
            for i in range(n):
                if nums[i] == num:
                    dist = abs(i - prev)
                    max_dist = max(max_dist, min(dist, n - dist))
                    prev = i
            
            ans = min(ans, (max_dist + 1) // 2)
        
        return ans