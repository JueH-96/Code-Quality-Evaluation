class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        array = []
        y = 1
        while len(array) < n:
            if y not in array:
                can_add = True
                for x in array:
                    if x + y == k:
                        can_add = False
                        break
                if can_add:
                    array.append(y)
            y += 1
        return sum(array)