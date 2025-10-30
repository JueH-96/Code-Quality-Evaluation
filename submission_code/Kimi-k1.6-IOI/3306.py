class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        sorted_nums = sorted(zip(nums, range(n)), key=lambda x: (x[0], x[1]))
        marked = [False] * n
        total = sum(nums)
        ptr = 0
        answer = []
        
        for index_i, k_i in queries:
            # Mark the specified index if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                total -= nums[index_i]
            
            # Mark up to k_i smallest unmarked elements
            remaining = k_i
            while remaining > 0 and ptr < len(sorted_nums):
                val, idx = sorted_nums[ptr]
                if not marked[idx]:
                    marked[idx] = True
                    total -= val
                    remaining -= 1
                ptr += 1
            
            answer.append(total)
        
        return answer