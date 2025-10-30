class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apple = sum(apple)
        sorted_cap = sorted(capacity, reverse=True)
        sum_cap = 0
        count = 0
        for c in sorted_cap:
            sum_cap += c
            count += 1
            if sum_cap >= total_apple:
                return count