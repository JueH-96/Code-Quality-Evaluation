class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        
        def check(x):
            chosen = []
            for i in range(n):
                chosen.append(start[i])
            
            
            for i in range(n):
                best = -1
                for j in range(start[i], start[i] + d + 1):
                    valid = True
                    for k in range(len(chosen)):
                        if k != i and abs(j - chosen[k]) < x:
                            valid = False
                            break
                    if valid:
                        chosen[i] = j
                        break

            
            for i in range(n):
                if chosen[i] == start[i]:
                    return False

            for i in range(n):
                for j in range(i + 1, n):
                    if abs(chosen[i] - chosen[j]) < x:
                        return False
            return True

        left = 1
        right = 2 * 10**9
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans