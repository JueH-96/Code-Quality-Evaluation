class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        maximumHeight.sort()
        heights = list(range(1, n + 1))
        total_sum = 0
        possible = True
        for i in range(n):
            if maximumHeight[i] < heights[i]:
                possible = False
                break
            total_sum += heights[i]
        
        if possible:
            
            ans = 0
            
            
            permutation = list(range(1,n+1))
            
            
            def solve(index,current_sum):
                nonlocal ans
                if index == n:
                    ans = max(ans,current_sum)
                    return
                
                for i in range(n):
                    if permutation[i] != -1 and permutation[i] <= maximumHeight[index]:
                        temp = permutation[i]
                        permutation[i] = -1
                        solve(index+1,current_sum + temp)
                        permutation[i] = temp
            
            solve(0,0)
            return ans
            
            
        else:
            return -1