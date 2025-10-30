from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        answer = []
        
        for index, k in queries:
            # Mark the element at index if it is not already marked
            if not marked[index]:
                marked[index] = True
            
            # Mark k unmarked elements with the smallest values
            unmarked = [(nums[i], i) for i in range(n) if not marked[i]]
            unmarked.sort()
            for _, i in unmarked[:k]:
                marked[i] = True
            
            # Calculate the sum of unmarked elements
            unmarked_sum = sum(nums[i] for i in range(n) if not marked[i])
            answer.append(unmarked_sum)
        
        return answer