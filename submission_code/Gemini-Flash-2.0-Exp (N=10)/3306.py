import heapq

class Solution:
    def unmarkedSumArray(self, nums: list[int], queries: list[list[int]]) -> list[int]:
        n = len(nums)
        marked = [False] * n
        result = []

        for index, k in queries:
            if not marked[index]:
                marked[index] = True
            
            unmarked_elements = []
            for i in range(n):
                if not marked[i]:
                    unmarked_elements.append((nums[i], i))
            
            heapq.heapify(unmarked_elements)
            
            count = 0
            while count < k and unmarked_elements:
                _, idx = heapq.heappop(unmarked_elements)
                if not marked[idx]:
                    marked[idx] = True
                    count += 1

            unmarked_sum = 0
            for i in range(n):
                if not marked[i]:
                    unmarked_sum += nums[i]
            
            result.append(unmarked_sum)
        
        return result