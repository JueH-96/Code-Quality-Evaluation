class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_nums = sorted([(num, i) for i, num in enumerate(nums)])
        pos = [0] * n
        for i in range(n):
            pos[sorted_nums[i][1]] = i
        bit = [0] * (n + 1)
        
        def update(i: int) -> None:
            i += 1
            while i <= n:
                bit[i] += 1
                i += i & -i

        def query(i: int) -> int:
            res = 0
            i += 1
            while i > 0:
                res += bit[i]
                i -= i & -i
            return res

        res = [0] * n
        for num, i in sorted_nums:
            left, right = 0, pos[i]
            while left < right:
                mid = (left + right) // 2
                if mid - query(mid) < pos[i] - query(pos[i]):
                    left = mid + 1
                else:
                    right = mid
            res[left] = num
            update(left)
        return res