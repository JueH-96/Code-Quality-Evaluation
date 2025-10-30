class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        
        n = len(nums)
        
        if n == 1:
            return nums[0]
        
        q = []
        for x in nums:
            q.append([x,1])
        
        
        max_val = 0
        
        while len(q) > 1:
            
            
            
            new_q = []
            
            i = 0
            while i < len(q) -1:
                if q[i][0] <= q[i+1][0]:
                    new_val = q[i][0] + q[i+1][0]
                    new_count = q[i][1] + q[i+1][1]
                    new_q.append([new_val,new_count])
                    i += 2
                else:
                    new_q.append(q[i])
                    i += 1
            if i < len(q):
                new_q.append(q[i])
            
            q = new_q
            
        return q[0][0]