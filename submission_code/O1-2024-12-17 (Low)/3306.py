class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        
        n = len(nums)
        # Keep track of which indices are marked
        marked = [False]*n
        
        # Min-heap of (nums[i], i), so we always have access to the smallest unmarked by value then index
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        # Keep track of total sum of all elements and the sum of marked elements
        total_sum = sum(nums)
        marked_sum = 0
        
        ans = []
        
        for index_i, k_i in queries:
            # Mark the element at index_i if it's not already marked
            if not marked[index_i]:
                marked[index_i] = True
                marked_sum += nums[index_i]
            
            # Mark k_i unmarked elements with the smallest values (and smallest indices if tie)
            to_mark = k_i
            while to_mark > 0 and heap:
                # Remove any elements at the top of the heap that are already marked
                while heap and marked[heap[0][1]]:
                    heapq.heappop(heap)
                if not heap:
                    break
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    marked_sum += val
                    to_mark -= 1
            
            ans.append(total_sum - marked_sum)
        
        return ans