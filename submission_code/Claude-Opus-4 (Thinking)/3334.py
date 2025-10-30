class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # Calculate total number of apples
        total_apples = sum(apple)
        
        # Sort boxes by capacity in descending order (largest first)
        capacity.sort(reverse=True)
        
        # Use boxes starting from largest capacity
        boxes_used = 0
        current_capacity = 0
        
        for cap in capacity:
            current_capacity += cap
            boxes_used += 1
            if current_capacity >= total_apples:
                return boxes_used
        
        return boxes_used