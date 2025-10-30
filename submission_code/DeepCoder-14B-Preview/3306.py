import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        total_sum = sum(nums)
        
        # Create a min-heap based on (value, index)
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        marked = set()
        marked_sum = 0
        answer = []
        
        for q in queries:
            index_i, k_i = q[0], q[1]
            
            # Mark the given index if not already marked
            if index_i not in marked:
                marked.add(index_i)
                marked_sum += nums[index_i]
            
            # Process the k_i elements
            count = 0
            while count < k_i:
                if not heap:
                    break
                value, idx = heapq.heappop(heap)
                if idx not in marked:
                    marked.add(idx)
                    marked_sum += value
                    count += 1
            
            # Calculate the current sum of unmarked elements
            current_unmarked = total_sum - marked_sum
            answer.append(current_unmarked)
        
        return answer