class Solution:
    def punishmentNumber(self, n: int) -> int:
        def check(num_str: str, target: int) -> bool:
            # DFS to try all partitions of num_str to sum to target
            length = len(num_str)
            def dfs(pos: int, curr_sum: int) -> bool:
                if pos == length:
                    return curr_sum == target
                # Early pruning
                if curr_sum > target:
                    return False
                # Try all possible next cuts
                val = 0
                for end in range(pos, length):
                    val = val * 10 + (ord(num_str[end]) - ord('0'))
                    if dfs(end + 1, curr_sum + val):
                        return True
                return False

            return dfs(0, 0)

        total = 0
        for i in range(1, n + 1):
            sq_str = str(i * i)
            if check(sq_str, i):
                total += i * i
        return total