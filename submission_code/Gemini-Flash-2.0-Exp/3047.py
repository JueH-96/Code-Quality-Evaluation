class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            
            current_sum = nums[i]
            visited[i] = True
            
            for j in range(i + 1, n):
                if not visited[j]:
                    product = nums[i] * nums[j]
                    
                    sqrt_product = int(product**0.5)
                    if sqrt_product * sqrt_product == product:
                        current_sum += nums[j]
                        visited[j] = True
            
            ans = max(ans, current_sum)
        
        return ans