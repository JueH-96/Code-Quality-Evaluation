class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        # Sort the box capacities in descending order
        capacity.sort(reverse=True)
        
        current_sum = 0
        box_count = 0
        
        # Accumulate until we reach or exceed the total apples
        for cap in capacity:
            current_sum += cap
            box_count += 1
            if current_sum >= total_apples:
                return box_count
        
        # This return should theoretically never be reached given the problem constraints
        return len(capacity)