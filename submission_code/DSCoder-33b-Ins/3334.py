class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        apple.sort(reverse=True)
        capacity.sort(reverse=True)
        i = j = 0
        while i < len(apple) and j < len(capacity):
            if apple[i] <= capacity[j]:
                i += 1
            j += 1
        return i