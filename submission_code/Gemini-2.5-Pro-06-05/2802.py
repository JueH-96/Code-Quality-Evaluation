class Solution:
    def punishmentNumber(self, n: int) -> int:
        
        total_sum = 0
        
        for i in range(1, n + 1):
            s = str(i * i)
            target = i
            
            # Memoization cache for the recursive calls for the current i.
            # State: (string_index, current_sum_of_partitions) -> bool
            memo = {}

            def dfs(index: int, current_sum: int) -> bool:
                # Check memoization cache to avoid recomputing for the same state.
                if (index, current_sum) in memo:
                    return memo[(index, current_sum)]

                # Base case: If we have reached the end of the string.
                # A valid partition is found if the sum of parts equals the target.
                if index == len(s):
                    return current_sum == target

                # Pruning: If the current sum already exceeds the target,
                # no valid partition can be found from this path.
                if current_sum > target:
                    return False

                # Recursive step: Try all possible next parts of the partition.
                res = False
                num = 0
                for j in range(index, len(s)):
                    num = num * 10 + int(s[j])
                    
                    # Optimization: If adding this part makes the sum too large,
                    # any larger part starting at the same index will also be too large.
                    if current_sum + num > target:
                        break
                    
                    # Recurse with the rest of the string and the updated sum.
                    if dfs(j + 1, current_sum + num):
                        res = True
                        break  # Found a valid partition, no need to check other possibilities.

                # Cache the result for this state and return it.
                memo[(index, current_sum)] = res
                return res

            # Start the recursive check for the current number i.
            if dfs(0, 0):
                total_sum += i * i
                
        return total_sum