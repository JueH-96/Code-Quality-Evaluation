class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        
        def check(score):
            chosen = [start[0]]
            for i in range(1, n):
                found = False
                for j in range(start[i], start[i] + d + 1):
                    valid = True
                    for c in chosen:
                        if abs(j - c) < score:
                            valid = False
                            break
                    if valid:
                        chosen.append(j)
                        found = True
                        break
                if not found:
                    return False
            return True
        
        left, right = 0, 10**9 + 1
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans