from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        sorted_with_indices = sorted(((nums[i], i) for i in range(n)), key=lambda x: (x[0], x[1]))
        pos_in_sorted = [0] * n
        for pos in range(n):
            _, original_index = sorted_with_indices[pos]
            pos_in_sorted[original_index] = pos
        
        parent = list(range(n))
        sum_unmarked = sum(nums)
        answer = []
        
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x: int) -> None:
            if x + 1 >= n:
                parent[x] = x + 1
            else:
                parent[x] = x + 1
        
        for query in queries:
            index_i, k_i = query
            pos = pos_in_sorted[index_i]
            root = find(pos)
            if root == pos:
                sum_unmarked -= nums[index_i]
                union(pos)
            
            count = 0
            while count < k_i:
                current_pos = find(0)
                if current_pos >= n:
                    break
                sum_unmarked -= sorted_with_indices[current_pos][0]
                union(current_pos)
                count += 1
            
            answer.append(sum_unmarked)
        
        return answer