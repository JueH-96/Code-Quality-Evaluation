class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        count = 0
        current_sum = 0
        
        for c in capacity:
            current_sum += c
            count += 1
            if current_sum >= total_apples:
                return count
        
        # The problem guarantees a solution exists, so we should never reach here.
        return -1