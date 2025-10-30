class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper to check if the square string can be partitioned to sum up to target
        def can_partition(s: str, target: int) -> bool:
            # DFS over split positions
            def dfs(pos: int, current_sum: int) -> bool:
                # If we've consumed all digits, check if we hit the target sum
                if pos == len(s):
                    return current_sum == target
                # Try every possible next cut
                for cut in range(pos + 1, len(s) + 1):
                    num = int(s[pos:cut])
                    new_sum = current_sum + num
                    # Prune if we already exceed the target
                    if new_sum > target:
                        continue
                    if dfs(cut, new_sum):
                        return True
                return False

            return dfs(0, 0)

        result = 0
        # For each i from 1 to n, check the punishment condition
        for i in range(1, n + 1):
            sq_str = str(i * i)
            if can_partition(sq_str, i):
                result += i * i

        return result