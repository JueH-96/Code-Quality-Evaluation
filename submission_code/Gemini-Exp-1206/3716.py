class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0

        def solve(idx, curr_seq):
            nonlocal ans
            if idx == n:
                if len(curr_seq) > 1:
                    diffs = []
                    for i in range(len(curr_seq) - 1):
                        diffs.append(abs(curr_seq[i+1] - curr_seq[i]))
                    
                    is_non_increasing = True
                    for i in range(len(diffs) - 1):
                        if diffs[i] < diffs[i+1]:
                            is_non_increasing = False
                            break
                    
                    if is_non_increasing:
                        ans = max(ans, len(curr_seq))
                return

            solve(idx + 1, curr_seq)
            solve(idx + 1, curr_seq + [nums[idx]])

        solve(0, [])
        return ans if ans > 0 else 1