class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        box_count = 0
        current_capacity = 0
        
        for cap in capacity:
            if current_capacity >= total_apples:
                break
            current_capacity += cap
            box_count += 1
        
        return box_count