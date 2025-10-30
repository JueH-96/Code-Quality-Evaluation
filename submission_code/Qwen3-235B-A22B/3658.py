from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def get_runs():
            runs = []
            n = len(nums)
            i = 0
            while i < n:
                if nums[i] == -1:
                    start = i
                    while i < n and nums[i] == -1:
                        i += 1
                    # start to i-1 is the run
                    L = nums[start - 1] if start > 0 else None
                    R = nums[i] if i < n else None
                    run_length = i - start
                    runs.append((L, R, run_length))
                else:
                    i += 1
            return runs

        # Compute base_max
        base_max = 0
        n = len(nums)
        for i in range(n - 1):
            if nums[i] != -1 and nums[i + 1] != -1:
                base_max = max(base_max, abs(nums[i] - nums[i + 1]))
        
        runs = get_runs()
        low = base_max
        high = 10**9
        answer = high
        
        while low <= high:
            mid = (low + high) // 2
            if base_max > mid:
                low = mid + 1
                continue
            
            # Generate candidates for x and y
            candidates = set()
            for (L, R, _) in runs:
                if L is not None:
                    candidates.add(L)
                    candidates.add(L - mid)
                    candidates.add(L + mid)
                if R is not None:
                    candidates.add(R)
                    candidates.add(R - mid)
                    candidates.add(R + mid)
            # Add some defaults
            candidates.update([1, 2, 10**9])
            cand_list = [x for x in candidates if x > 0]
            if not cand_list:
                cand_list = [1]
            
            possible = False
            # Try all pairs of candidates
            for x in cand_list:
                for y in cand_list:
                    valid = True
                    for (L, R, run_len) in runs:
                        # Check run conditions
                        run_ok = False
                        if run_len == 1:
                            # Run of length 1: filled with x or y must satisfy both L and R
                            cond_x = True
                            if L is not None:
                                cond_x &= abs(x - L) <= mid
                            if R is not None:
                                cond_x &= abs(x - R) <= mid
                            cond_y = True
                            if L is not None:
                                cond_y &= abs(y - L) <= mid
                            if R is not None:
                                cond_y &= abs(y - R) <= mid
                            run_ok = cond_x or cond_y
                        else:
                            # Run of length >= 2
                            if abs(x - y) > mid:
                                # Must fill entire run with x or y
                                cond_x = True
                                if L is not None:
                                    cond_x &= abs(x - L) <= mid
                                if R is not None:
                                    cond_x &= abs(x - R) <= mid
                                cond_y = True
                                if L is not None:
                                    cond_y &= abs(y - L) <= mid
                                if R is not None:
                                    cond_y &= abs(y - R) <= mid
                                run_ok = cond_x or cond_y
                            else:
                                # Can choose x/y for ends
                                left_ok = True
                                if L is not None:
                                    left_ok = (abs(x - L) <= mid) or (abs(y - L) <= mid)
                                right_ok = True
                                if R is not None:
                                    right_ok = (abs(x - R) <= mid) or (abs(y - R) <= mid)
                                run_ok = left_ok and right_ok
                        if not run_ok:
                            valid = False
                            break
                    if valid:
                        possible = True
                        break
                if possible:
                    break
            if possible:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer