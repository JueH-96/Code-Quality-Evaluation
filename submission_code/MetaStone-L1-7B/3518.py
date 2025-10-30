class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        first = second = third = fourth = float('-inf')
        for num in b:
            new_first = max(first, a[0] * num)
            new_second = max(second, a[1] * num + first)
            new_third = max(third, a[2] * num + second)
            new_fourth = max(fourth, a[3] * num + third)
            first, second, third, fourth = new_first, new_second, new_third, new_fourth
        return fourth