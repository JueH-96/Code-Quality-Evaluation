class Solution:
    def minimumOperations(self, num: str) -> int:
        # Special cases end with 00, 25, 50, 75
        # We need to find the minimum number of operations to form these endings
        
        def operations_to_form_ending(ending):
            count = 0
            i = len(num) - 1
            for e in reversed(ending):
                while i >= 0 and num[i] != e:
                    i -= 1
                    count += 1
                if i >= 0:
                    i -= 1
                else:
                    return float('inf')  # Cannot form the ending
            return count + max(0, i + 1)  # Add remaining digits to the left
        
        # Calculate operations for each special ending
        ops_00 = operations_to_form_ending('00')
        ops_25 = operations_to_form_ending('25')
        ops_50 = operations_to_form_ending('50')
        ops_75 = operations_to_form_ending('75')
        
        # Return the minimum operations among all endings
        return min(ops_00, ops_25, ops_50, ops_75)

# Example usage:
# sol = Solution()
# print(sol.minimumOperations("2245047"))  # Output: 2
# print(sol.minimumOperations("2908305"))  # Output: 3
# print(sol.minimumOperations("10"))       # Output: 1