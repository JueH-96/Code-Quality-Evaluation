class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        def count_operations(x: int, y: int) -> int:
            operations = 0
            while x != y:
                if x > y:
                    x -= y
                else:
                    y -= x
                operations += 1
            return operations

        return min(count_operations(x, y), count_operations(y, x))