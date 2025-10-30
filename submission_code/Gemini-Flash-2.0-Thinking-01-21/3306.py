class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        answer = []
        for query in queries:
            index_i, k_i = query
            if not marked[index_i]:
                marked[index_i] = True
            
            unmarked_elements = []
            for j in range(n):
                if not marked[j]:
                    unmarked_elements.append((nums[j], j))
            
            unmarked_elements.sort(key=lambda x: (x[0], x[1]))
            
            num_to_mark = min(k_i, len(unmarked_elements))
            for i in range(num_to_mark):
                mark_index = unmarked_elements[i][1]
                marked[mark_index] = True
                
            unmarked_sum = 0
            for j in range(n):
                if not marked[j]:
                    unmarked_sum += nums[j]
            answer.append(unmarked_sum)
        return answer