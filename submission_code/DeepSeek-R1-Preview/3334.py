class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        current_capacity = 0
        box_count = 0
        
        for cap in capacity:
            current_capacity += cap
            box_count += 1
            if current_capacity >= total_apples:
                return box_count
        
        return box_count  # This line is theoretically unreachable due to problem constraints