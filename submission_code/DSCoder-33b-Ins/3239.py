class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        elif x > y:
            return self.minimumOperationsToMakeEqual(x - y, y) + 1
        else:
            return self.minimumOperationsToMakeEqual(y - x, x) + 1