class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        import heapq
        from sortedcontainers import SortedList
        
        n = len(nums)
        m = len(queries)
        marked = [False] * n
        unmarked_values = SortedList(nums)
        answer = []
        
        for index, k in queries:
            # Mark the element at index if not already marked
            if not marked[index]:
                marked[index] = True
                unmarked_values.remove(nums[index])
            
            # Mark k smallest unmarked elements
            count = 0
            while count < k and unmarked_values:
                min_val = unmarked_values[0]
                # Remove all occurrences of min_val from unmarked_values up to k-count times
                occurrences = min(unmarked_values.count(min_val), k - count)
                for _ in range(occurrences):
                    unmarked_values.remove(min_val)
                count += occurrences
            
            # Calculate the sum of unmarked elements
            answer.append(sum(unmarked_values))
        
        return answer