from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        
        # Step 1: Sort the nums with their indices
        sorted_nums = sorted([(nums[i], i) for i in range(n)], key=lambda x: (x[0], x[1]))
        
        # Step 2: Initialize marked array and total_sum
        marked = [False] * n
        total_sum = sum(nums)
        
        # Step 3: Initialize pointer
        ptr = 0
        answer = []
        
        # Step 4: Process each query
        for query in queries:
            index_i, k_i = query
            # Mark the specified index if not marked
            if not marked[index_i]:
                marked[index_i] = True
                total_sum -= nums[index_i]
            # Mark k_i smallest unmarked elements
            while ptr < n and k_i > 0:
                if not marked[sorted_nums[ptr][1]]:
                    marked[sorted_nums[ptr][1]] = True
                    total_sum -= sorted_nums[ptr][0]
                    k_i -= 1
                ptr += 1
            # Append the current total_sum to the answer
            answer.append(total_sum)
        
        return answer