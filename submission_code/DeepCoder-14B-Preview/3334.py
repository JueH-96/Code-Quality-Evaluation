class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        sorted_capacity = sorted(capacity, reverse=True)
        current_sum = 0
        boxes_used = 0
        
        for cap in sorted_capacity:
            current_sum += cap
            boxes_used += 1
            if current_sum >= total_apples:
                break
        
        return boxes_used