from typing import List
from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        N = len(nums)
        index_map = defaultdict(list)
        for i in range(N):
            index_map[nums[i]].append(i)
        
        ans = [-1] * N
        
        for pos_list in index_map.values():
            M = len(pos_list)
            if M > 1:
                for k in range(M):
                    left_k = (k - 1) % M
                    right_k = (k + 1) % M
                    left_pos = pos_list[left_k]
                    right_pos = pos_list[right_k]
                    diff_left = abs(pos_list[k] - left_pos)
                    dist_left = min(diff_left, N - diff_left)
                    diff_right = abs(pos_list[k] - right_pos)
                    dist_right = min(diff_right, N - diff_right)
                    min_dist = min(dist_left, dist_right)
                    idx = pos_list[k]
                    ans[idx] = min_dist
        
        return [ans[q] for q in queries]