class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        n, m = len(apple), len(capacity)
        capacity.sort(reverse=True)
        
        total_apples = sum(apple)
        boxes_used = 0
        
        for box_capacity in capacity:
            if total_apples <= 0:
                break
            
            boxes_used += 1
            total_apples -= box_capacity
        
        return boxes_used