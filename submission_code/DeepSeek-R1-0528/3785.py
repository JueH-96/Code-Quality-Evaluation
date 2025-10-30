class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        global_lower = bounds[0][0]
        global_upper = bounds[0][1]
        for i in range(1, n):
            offset = original[i] - original[0]
            low_bound = bounds[i][0] - offset
            high_bound = bounds[i][1] - offset
            global_lower = max(global_lower, low_bound)
            global_upper = min(global_upper, high_bound)
            if global_lower > global_upper:
                return 0
        return global_upper - global_lower + 1