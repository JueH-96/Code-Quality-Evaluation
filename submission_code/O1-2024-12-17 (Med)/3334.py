class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        # Sort boxes in descending order of their capacity
        capacity.sort(reverse=True)
        
        current_capacity_sum = 0
        box_count = 0
        
        for cap in capacity:
            current_capacity_sum += cap
            box_count += 1
            if current_capacity_sum >= total_apples:
                break
        
        return box_count