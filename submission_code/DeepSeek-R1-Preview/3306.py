class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        sorted_elements = sorted([(nums[i], i) for i in range(n)], key=lambda x: (x[0], x[1]))
        marked = [False] * n
        total = sum(nums)
        current_marked_sum = 0
        res = []
        current = 0  # Pointer to track the next element to consider in sorted_elements
        
        for idx, k in queries:
            # Step a: Mark the index if not marked
            if not marked[idx]:
                marked[idx] = True
                current_marked_sum += nums[idx]
            
            # Step b: Mark up to k elements from the sorted_elements
            count = 0
            while current < len(sorted_elements) and count < k:
                num, i = sorted_elements[current]
                if not marked[i]:
                    marked[i] = True
                    current_marked_sum += num
                    count += 1
                current += 1  # Move to next element regardless of whether it was marked
            
            res.append(total - current_marked_sum)
        
        return res