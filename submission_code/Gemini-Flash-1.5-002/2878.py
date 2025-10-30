class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        if n == 0:
            return True

        q = deque()
        for i in range(k):
            q.append(nums[i])

        
        for i in range(k,n):
            
            mx = max(q)
            
            
            if mx > sum(q) - mx:
                return False
            
            
            q.popleft()
            q.append(nums[i])

        mx = max(q)
        if mx > sum(q) - mx:
            return False

        return True