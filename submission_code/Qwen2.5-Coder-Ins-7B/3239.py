class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x <= y:
            return y - x
        else:
            return 1 + min(self.minimumOperationsToMakeEqual(x // 11, y), self.minimumOperationsToMakeEqual(x // 5, y), self.minimumOperationsToMakeEqual(x + 1, y), self.minimumOperationsToMakeEqual(x - 1, y))