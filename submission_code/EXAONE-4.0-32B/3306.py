class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        
        sorted_arr = [(nums[i], i) for i in range(n)]
        sorted_arr.sort(key=lambda x: (x[0], x[1]))
        
        pos = [0] * n
        for idx, (_, orig_idx) in enumerate(sorted_arr):
            pos[orig_idx] = idx
        
        next_ptr = [0] * n
        prev = [0] * n
        for i in range(n):
            if i == 0:
                prev[i] = -1
            else:
                prev[i] = i - 1
            if i == n - 1:
                next_ptr[i] = -1
            else:
                next_ptr[i] = i + 1
        
        cur = 0
        unmarked_sum = sum(nums)
        marked = [False] * n
        ans = [0] * m
        
        for i in range(m):
            idx = queries[i][0]
            k = queries[i][1]
            
            if not marked[idx]:
                marked[idx] = True
                p = pos[idx]
                unmarked_sum -= nums[idx]
                
                if prev[p] != -1:
                    next_ptr[prev[p]] = next_ptr[p]
                else:
                    cur = next_ptr[p]
                
                if next_ptr[p] != -1:
                    prev[next_ptr[p]] = prev[p]
            
            count = 0
            temp = cur
            while count < k and temp != -1:
                val, orig_idx = sorted_arr[temp]
                if marked[orig_idx]:
                    temp = next_ptr[temp]
                    continue
                    
                marked[orig_idx] = True
                unmarked_sum -= val
                count += 1
                
                if prev[temp] != -1:
                    next_ptr[prev[temp]] = next_ptr[temp]
                else:
                    cur = next_ptr[temp]
                
                if next_ptr[temp] != -1:
                    prev[next_ptr[temp]] = prev[temp]
                
                temp = next_ptr[temp]
            
            ans[i] = unmarked_sum
        
        return ans