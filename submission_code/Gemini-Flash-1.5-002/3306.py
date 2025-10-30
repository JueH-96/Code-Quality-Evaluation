class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        marked = [False] * n
        answer = []
        
        for index, k in queries:
            if not marked[index]:
                marked[index] = True
            
            unmarked_indices = []
            for i in range(n):
                if not marked[i]:
                    unmarked_indices.append(i)
            
            unmarked_indices.sort(key=lambda i: nums[i])
            
            count = 0
            for i in range(min(k, len(unmarked_indices))):
                marked[unmarked_indices[i]] = True
                count +=1

            unmarked_sum = 0
            for i in range(n):
                if not marked[i]:
                    unmarked_sum += nums[i]
            answer.append(unmarked_sum)
        return answer