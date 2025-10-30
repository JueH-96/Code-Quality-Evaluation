class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)  # Sort boxes by capacity in descending order
        
        boxes_needed = 0
        total_capacity = 0
        
        for cap in capacity:
            boxes_needed += 1
            total_capacity += cap
            if total_capacity >= total_apples:
                return boxes_needed
        
        return boxes_needed