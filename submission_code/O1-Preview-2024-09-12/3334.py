class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacities = sorted(capacity, reverse=True)
        total_capacity = 0
        n_boxes = 0
        for c in capacities:
            total_capacity += c
            n_boxes += 1
            if total_capacity >= total_apples:
                return n_boxes