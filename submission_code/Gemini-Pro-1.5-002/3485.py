class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        intervals = []
        for i in range(n):
            intervals.append((start[i], start[i] + d))
        
        intervals.sort()

        def check(score):
            chosen = []
            for l, r in intervals:
                valid_choice = False
                for choice in range(l, r + 1):
                    is_valid = True
                    for prev_choice in chosen:
                        if abs(choice - prev_choice) < score:
                            is_valid = False
                            break
                    if is_valid:
                        chosen.append(choice)
                        valid_choice = True
                        break
                if not valid_choice:
                    return False
            return True

        left, right = 0, 2 * 10**9
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans