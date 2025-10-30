class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        arr = []
        x = 1
        while len(arr) < n:
            conflict = False
            for num in arr:
                if num + x == k:
                    conflict = True
                    break
            if not conflict:
                arr.append(x)
            x += 1
        return sum(arr)