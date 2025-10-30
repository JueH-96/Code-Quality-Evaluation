class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        ans = []

        for index, k in queries:
            if not marked[index]:
                marked[index] = True

            unmarked_indices = []
            for i in range(n):
                if not marked[i]:
                    unmarked_indices.append(i)
            
            sorted_unmarked_indices = sorted(unmarked_indices, key=lambda x: nums[x])
            
            count = 0
            for i in sorted_unmarked_indices:
                if count < k:
                    marked[i] = True
                    count += 1
                else:
                    break
            
            current_sum = 0
            for i in range(n):
                if not marked[i]:
                    current_sum += nums[i]
            ans.append(current_sum)
        
        return ans