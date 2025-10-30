class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        cur_M = [-1] * n
        res_arr = []
        
        dp_prev = [nums[0]]
        cur_M[0] = nums[0]
        res_arr.append([nums[0]])
        
        for j in range(1, n):
            current = [0] * (j+1)
            current[j] = nums[j]
            if nums[j] > cur_M[j]:
                cur_M[j] = nums[j]
            for i in range(j-1, -1, -1):
                current[i] = dp_prev[i] ^ current[i+1]
                if current[i] > cur_M[i]:
                    cur_M[i] = current[i]
            dp_prev = current
            
            arr = [0] * (j+1)
            cur_max = -1
            for i in range(j, -1, -1):
                if cur_M[i] > cur_max:
                    cur_max = cur_M[i]
                arr[i] = cur_max
            res_arr.append(arr)
        
        ans = []
        for l, r in queries:
            ans.append(res_arr[r][l])
        return ans