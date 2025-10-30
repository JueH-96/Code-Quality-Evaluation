class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import sys
        input_data = sys.stdin.readline
        
        n = len(nums)
        # Pair each number with its index and sort by (value, index)
        sorted_value_index = sorted([(val, idx) for idx, val in enumerate(nums)], key=lambda x: (x[0], x[1]))
        
        # This will track whether an element is marked
        marked = [False]*n
        
        # Current pointer into the sorted list
        cur = 0
        
        # Track the sum of unmarked elements
        sum_unmarked = sum(nums)
        
        answer = []
        
        for index_i, k_i in queries:
            # 1) Mark the element at index_i if it's not already marked
            if not marked[index_i]:
                marked[index_i] = True
                sum_unmarked -= nums[index_i]
            
            # 2) Mark k_i unmarked elements in ascending order of (value, index)
            while k_i > 0 and cur < n:
                val, idx = sorted_value_index[cur]
                if not marked[idx]:
                    marked[idx] = True
                    sum_unmarked -= val
                    k_i -= 1
                cur += 1
            
            # The sum of unmarked elements after this query
            answer.append(sum_unmarked)
        
        return answer