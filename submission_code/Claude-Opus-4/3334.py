class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # Calculate total number of apples
        total_apples = sum(apple)
        
        # Sort boxes by capacity in descending order
        capacity.sort(reverse=True)
        
        # Use boxes starting from the largest
        boxes_used = 0
        current_capacity = 0
        
        for box_capacity in capacity:
            current_capacity += box_capacity
            boxes_used += 1
            
            # If we have enough capacity, return the number of boxes used
            if current_capacity >= total_apples:
                return boxes_used
        
        # This line should never be reached given the constraints
        return boxes_used