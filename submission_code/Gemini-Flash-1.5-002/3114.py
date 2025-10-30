class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        ans = 0
        for i in range(n):
            left = [1] * n
            right = [1] * n
            for j in range(1, n):
                if maxHeights[j] >= maxHeights[j - 1]:
                    left[j] = left[j - 1] + 1
                if maxHeights[n - j - 1] >= maxHeights[n - j]:
                    right[n - j - 1] = right[n - j] + 1

            cur = 0
            for j in range(n):
                h = min(maxHeights[j], left[j], right[j])
                cur += h
            ans = max(ans, cur)

            for j in range(n):
                
                min_h = min(maxHeights[j],left[j],right[j])
                
                cur_sum = 0
                temp_heights = [0]*n
                
                for k in range(j+1):
                    temp_heights[k] = min(maxHeights[k],min_h)
                    
                for k in range(j+1,n):
                    temp_heights[k] = min(maxHeights[k],min_h)
                    
                
                
                is_mountain = True
                peak_index = j
                
                for k in range(1,peak_index+1):
                    if temp_heights[k] < temp_heights[k-1]:
                        is_mountain = False
                        break
                if is_mountain == False:
                    continue
                for k in range(peak_index,n-1):
                    if temp_heights[k] < temp_heights[k+1]:
                        is_mountain = False
                        break
                if is_mountain == False:
                    continue
                
                cur_sum = sum(temp_heights)
                ans = max(ans,cur_sum)

        return ans