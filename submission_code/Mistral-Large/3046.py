class Solution:
    def minimumOperations(self, num: str) -> int:
        targets = ["00", "25", "50", "75"]
        n = len(num)
        min_operations = float('inf')

        for target in targets:
            t1, t2 = target
            operations = 0
            i = n - 1

            while i >= 0 and operations < min_operations:
                if num[i] == t2:
                    j = i - 1
                    while j >= 0 and operations < min_operations:
                        if num[j] == t1:
                            min_operations = operations
                            break
                        j -= 1
                        operations += 1
                i -= 1
                operations += 1

        return min_operations

# Example usage:
# solution = Solution()
# print(solution.minimumOperations("2245047"))  # Output: 2
# print(solution.minimumOperations("2908305"))  # Output: 3
# print(solution.minimumOperations("10"))      # Output: 1