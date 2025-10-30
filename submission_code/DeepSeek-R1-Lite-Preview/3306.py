from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        # Sort elements by value and then by index
        sorted_elements = sorted([(nums[i], i) for i in range(n)], key=lambda x: (x[0], x[1]))
        marked = set()
        total_sum = sum(nums)
        ptr = 0
        answer = []
        
        for query in queries:
            index_i, k_i = query
            # Mark the specified index if not marked
            if index_i not in marked:
                marked.add(index_i)
                total_sum -= nums[index_i]
            # Mark the k_i smallest unmarked elements
            while k_i > 0 and ptr < n:
                value, idx = sorted_elements[ptr]
                if idx not in marked:
                    marked.add(idx)
                    total_sum -= value
                    k_i -= 1
                ptr += 1
            answer.append(total_sum)
        
        return answer