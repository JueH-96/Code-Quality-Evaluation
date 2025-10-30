class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        start_sorted = sorted(start)

        def is_possible_score(score):
            chosen_values = []
            current_val = start_sorted[0]
            chosen_values.append(current_val)
            for i in range(1, n):
                next_val_min = chosen_values[-1] + score
                if next_val_min > start_sorted[i] + d:
                    return False
                current_val = max(start_sorted[i], next_val_min)
                if current_val > start_sorted[i] + d:
                    return False
                chosen_values.append(current_val)
            return True

        low = 0
        high = 2 * 10**9
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if is_possible_score(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans