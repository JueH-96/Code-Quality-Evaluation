class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        apple.sort(reverse=True)
        capacity.sort(reverse=True)

        total_apples = sum(apple)
        total_boxes = 0
        box_count = 0

        while total_apples > 0:
            if total_boxes + capacity[total_boxes] >= total_apples:
                total_apples -= apple[total_boxes]
                total_boxes += 1
                box_count += 1
            else:
                total_boxes += 1

        return box_count