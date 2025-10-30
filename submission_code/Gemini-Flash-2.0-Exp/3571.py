class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        
        def solve(start_index):
            dp = {}

            def find_path(curr_index):
                if curr_index in dp:
                    return dp[curr_index]

                max_len = 1
                for next_index in range(n):
                    if coordinates[next_index][0] > coordinates[curr_index][0] and \
                       coordinates[next_index][1] > coordinates[curr_index][1]:
                        max_len = max(max_len, 1 + find_path(next_index))

                dp[curr_index] = max_len
                return max_len

            return find_path(start_index)

        
        return solve(k)