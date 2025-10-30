class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0

        # Calculate the size of the segment tree
        self.size = 1
        while self.size < n:
            self.size <<= 1
        self.size <<= 1

        self.tree = [ (0, 0, 0, 0) for _ in range(self.size) ]

        def build(node, start, end):
            if start == end:
                x = nums[start]
                a = x
                b = -float('inf')
                c = -float('inf')
                d = 0
                self.tree[node] = (a, b, c, d)
            else:
                mid = (start + end) // 2
                build(2*node, start, mid)
                build(2*node + 1, mid + 1, end)
                left = self.tree[2*node]
                right = self.tree[2*node + 1]

                a = max(
                    left[0] + right[2],
                    left[1] + right[0],
                    left[1] + right[2]
                ) if any([left[0] + right[2] != -float('inf'), left[1] + right[0] != -float('inf'), left[1] + right[2] != -float('inf')]) else -float('inf')

                b = max(
                    left[0] + right[3],
                    left[1] + right[1],
                    left[1] + right[3]
                ) if any([left[0] + right[3] != -float('inf'), left[1] + right[1] != -float('inf'), left[1] + right[3] != -float('inf')]) else -float('inf')

                c = max(
                    left[2] + right[2],
                    left[3] + right[0],
                    left[3] + right[2]
                ) if any([left[2] + right[2] != -float('inf'), left[3] + right[0] != -float('inf'), left[3] + right[2] != -float('inf')]) else -float('inf')

                d_val = max(
                    left[2] + right[3],
                    left[3] + right[1],
                    left[3] + right[3]
                ) if any([left[2] + right[3] != -float('inf'), left[3] + right[1] != -float('inf'), left[3] + right[3] != -float('inf')]) else -float('inf')

                self.tree[node] = (a, b, c, d_val)

        build(1, 0, n-1)

        def update(pos, value):
            stack = []
            node = 1
            start, end = 0, n-1
            while start != end:
                mid = (start + end) // 2
                stack.append(node)
                if pos <= mid:
                    node = 2 * node
                    end = mid
                else:
                    node = 2 * node + 1
                    start = mid + 1

            a = value
            b = -float('inf')
            c = -float('inf')
            d = 0
            self.tree[node] = (a, b, c, d)

            while stack:
                node = stack.pop()
                left_node = 2 * node
                right_node = 2 * node + 1

                left = self.tree[left_node]
                right = self.tree[right_node]

                a = max(
                    left[0] + right[2],
                    left[1] + right[0],
                    left[1] + right[2]
                ) if any([left[0] + right[2] != -float('inf'), left[1] + right[0] != -float('inf'), left[1] + right[2] != -float('inf')]) else -float('inf')

                b = max(
                    left[0] + right[3],
                    left[1] + right[1],
                    left[1] + right[3]
                ) if any([left[0] + right[3] != -float('inf'), left[1] + right[1] != -float('inf'), left[1] + right[3] != -float('inf')]) else -float('inf')

                c = max(
                    left[2] + right[2],
                    left[3] + right[0],
                    left[3] + right[2]
                ) if any([left[2] + right[2] != -float('inf'), left[3] + right[0] != -float('inf'), left[3] + right[2] != -float('inf')]) else -float('inf')

                d_val = max(
                    left[2] + right[3],
                    left[3] + right[1],
                    left[3] + right[3]
                ) if any([left[2] + right[3] != -float('inf'), left[3] + right[1] != -float('inf'), left[3] + right[3] != -float('inf')]) else -float('inf')

                self.tree[node] = (a, b, c, d_val)

        res = 0
        for pos, x in queries:
            update(pos, x)
            a, b, c, d = self.tree[1]
            current_max = max(a, b, c, d)
            res = (res + max(current_max, 0)) % MOD

        return res % MOD