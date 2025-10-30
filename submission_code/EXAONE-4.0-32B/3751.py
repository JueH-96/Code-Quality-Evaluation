class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        base = nums.count(k)
        n = len(nums)
        ans = base
        
        for v in range(1, 51):
            arr = []
            if v == k:
                arr = [0] * n
            else:
                for num in nums:
                    if num == v:
                        arr.append(1)
                    elif num == k:
                        arr.append(-1)
                    else:
                        arr.append(0)
            
            if n == 0:
                candidate_val = 0
            else:
                current = arr[0]
                best_arr = arr[0]
                for i in range(1, n):
                    current = max(arr[i], current + arr[i])
                    best_arr = max(best_arr, current)
                candidate_val = best_arr
            
            total_candidate = base + candidate_val
            if total_candidate > ans:
                ans = total_candidate
                
        return ans