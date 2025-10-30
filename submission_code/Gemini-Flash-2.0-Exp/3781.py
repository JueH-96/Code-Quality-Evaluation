class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def check(dist):
            selected = [points[0]]
            for i in range(1, len(points)):
                valid = True
                for s in selected:
                    if abs(points[i][0] - s[0]) + abs(points[i][1] - s[1]) < dist:
                        valid = False
                        break
                if valid:
                    selected.append(points[i])
            return len(selected) >= k

        points.sort()
        left, right = 0, 2 * side
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans