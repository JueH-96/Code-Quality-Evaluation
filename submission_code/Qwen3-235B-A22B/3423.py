from typing import List
import sys
sys.setrecursionlimit(1 << 25)

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7

        def merge(left_dp, right_dp):
            merged = [[float('-inf')] * 2 for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    best = float('-inf')
                    for left_end in [0, 1]:
                        for right_start in [0, 1]:
                            if left_end + right_start <= 1:
                                le = left_dp[i][left_end]
                                ri = right_dp[right_start][j]
                                if le == float('-inf') or ri == float('-inf'):
                                    continue
                                current_sum = le + ri
                                if current_sum > best:
                                    best = current_sum
                    if best != float('-inf'):
                        merged[i][j] = best
                    else:
                        merged[i][j] = float('-inf')
            return merged

        def build(l, r):
            node = {'left': None, 'right': None, 'start': l, 'end': r, 'dp': [[float('-inf')] * 2 for _ in range(2)]}
            if l == r:
                val = nums[l]
                node['dp'][0][0] = 0
                node['dp'][0][1] = float('-inf')
                node['dp'][1][0] = float('-inf')
                node['dp'][1][1] = max(0, val)
            else:
                mid = (l + r) // 2
                node['left'] = build(l, mid)
                node['right'] = build(mid + 1, r)
                node['dp'] = merge(node['left']['dp'], node['right']['dp'])
            return node

        def update(node, pos, value):
            l = node['start']
            r = node['end']
            if l == r:
                val = value
                node['dp'][0][0] = 0
                node['dp'][0][1] = float('-inf')
                node['dp'][1][0] = float('-inf')
                node['dp'][1][1] = max(0, val)
            else:
                mid = (l + r) // 2
                if pos <= mid:
                    update(node['left'], pos, value)
                else:
                    update(node['right'], pos, value)
                node['dp'] = merge(node['left']['dp'], node['right']['dp'])

        root = build(0, len(nums) - 1)
        res = 0
        for pos, x in queries:
            update(root, pos, x)
            current_max = max(max(row) for row in root['dp'])
            res = (res + current_max) % MOD
        return res % MOD