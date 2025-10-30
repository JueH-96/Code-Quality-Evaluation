from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        # ---------- Li Chao Segment Tree (max variant) ----------
        class Node:
            __slots__ = ('l', 'r', 'line', 'left', 'right')
            def __init__(self, l: int, r: int):
                self.l = l
                self.r = r
                self.line = None          # (m, b)
                self.left: 'Node' = None
                self.right: 'Node' = None

        class LiChao:
            def __init__(self, x_left: int, x_right: int):
                self.root = Node(x_left, x_right)

            @staticmethod
            def _value(line, x: int) -> int:
                m, b = line
                return m * x + b

            # insert new line (m, b)
            def add_line(self, m: int, b: int) -> None:
                line = (m, b)
                self._add(self.root, line)

            def _add(self, node: Node, line) -> None:
                l, r = node.l, node.r
                mid = (l + r) // 2

                if node.line is None:
                    node.line = line
                    return

                # swap if new line is better in the middle
                if self._value(line, mid) > self._value(node.line, mid):
                    node.line, line = line, node.line

                if l == r:
                    return

                # decide which side might still need the worse line
                if self._value(line, l) > self._value(node.line, l):
                    if node.left is None:
                        node.left = Node(l, mid)
                    self._add(node.left, line)
                elif self._value(line, r) > self._value(node.line, r):
                    if node.right is None:
                        node.right = Node(mid + 1, r)
                    self._add(node.right, line)

            # query maximum value at x
            def query(self, x: int) -> int:
                return self._query(self.root, x)

            def _query(self, node: Node, x: int) -> int:
                if node is None:
                    return -10 ** 30          # effectively -INF
                res = self._value(node.line, x) if node.line else -10 ** 30
                if node.l == node.r:
                    return res
                mid = (node.l + node.r) // 2
                if x <= mid:
                    return max(res, self._query(node.left, x))
                else:
                    return max(res, self._query(node.right, x))

        # ----------------- main DP using Li Chao -----------------
        n = len(nums)
        if n == 1:                          # already at last index
            return 0

        max_x = max(nums)
        hull = LiChao(0, max_x)            # domain of x-values

        dp = [0] * n                       # dp[i] – best score from i to end
        dp[-1] = 0

        # insert line for the last position j = n-1 : y = (n-1)*x + dp[n-1]
        hull.add_line(n - 1, dp[-1])

        # process indices from right to left
        for i in range(n - 2, -1, -1):
            x = nums[i]
            best = hull.query(x)           # max_{j>i}  (j * x + dp[j])
            dp[i] = best - i * x           # = -i*x + best
            hull.add_line(i, dp[i])        # add line for current index

        return dp[0]