class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        import math

        n = len(nums)
        S = sum(nums)
        
        # If the sum of the entire array is exactly target, we might still find
        # a smaller sub-subarray within one copy (if such subarray also sums to S).
        # So we do not return n immediately; we'll rely on the generic logic.
        
        # Helper: find the minimal-length subarray in up to 2 copies of nums that sums to X.
        # Return None if no such subarray exists.
        def min_subarray_sum_in_2(X):
            if X == 0:
                return 0  # sum 0 would mean empty, but problem context needs positive sum, so won't be used this way
            arr = nums + nums
            left = 0
            cur_sum = 0
            min_len = math.inf
            for right, val in enumerate(arr):
                cur_sum += val
                # Shrink from the left while we can
                while cur_sum >= X:
                    if cur_sum == X:
                        min_len = min(min_len, right - left + 1)
                    cur_sum -= arr[left]
                    left += 1
                # If our window length exceeds n, there's no need to let it exceed 2*n,
                # but the loop is up to 2n anyway.
            return min_len if min_len != math.inf else None

        # Helper: find the minimal prefix length in up to 2 copies of nums that sums to X.
        # Return None if no such prefix exists.
        def min_prefix_sum_in_2(X):
            arr = nums + nums
            prefix_sum = 0
            for i, val in enumerate(arr):
                prefix_sum += val
                if prefix_sum == X:
                    return i + 1
            return None

        # Helper: find the minimal suffix length in up to 2 copies of nums that sums to X.
        # Return None if no such suffix exists.
        def min_suffix_sum_in_2(X):
            arr = nums + nums
            total = sum(arr)
            # We'll do a prefix-sum style approach from the left to find suffix = X
            # suffix sum from i..(2n-1) = total - prefix_sum_of_(i-1)
            prefix_sum = 0
            prefix_sums = [0]*(2*n + 1)
            for i in range(2*n):
                prefix_sum += arr[i]
                prefix_sums[i+1] = prefix_sum
            
            min_len = math.inf
            # We want total - prefix_sums[i] == X => prefix_sums[i] == total - X
            need = total - X
            # We'll look for all i such that prefix_sums[i] = need
            # Then suffix length is (2n - i).
            # We'll pick the smallest (2n - i).
            from collections import defaultdict
            pos_map = defaultdict(list)
            for i, ps in enumerate(prefix_sums):
                pos_map[ps].append(i)
            
            if need in pos_map:
                for i in pos_map[need]:
                    # i can range from 0..2n
                    if i <= 2*n:
                        length = (2*n - i)
                        if length > 0 and length < min_len:
                            min_len = length
            
            return min_len if min_len != math.inf else None
        
        # If sum of nums is less than target, we may need multiple copies,
        # otherwise possibly zero or one copy is enough. We'll use M = target//S, R = target%S
        # Then consider two main candidates:
        #   candidate1 = M*n + (min length subarray in 2 copies summing to R)  if R != 0
        #                else = M*n if R == 0
        #   candidate2 = (M+1)*n - (minimal prefix/suffix in 2 copies that sums to (M+1)*S - target)
        # We take the min of these two if they exist, otherwise -1.

        M = target // S
        R = target % S
        
        # Candidate 1
        if R == 0:
            # sum = M*S exactly
            # That means an entire subarray of M copies can achieve sum=target with length M*n
            candidate1 = M * n
        else:
            # Need an extra partial chunk that sums to R
            extra_len = min_subarray_sum_in_2(R)
            if extra_len is None:
                candidate1 = math.inf
            else:
                candidate1 = M * n + extra_len
        
        # Candidate 2
        # We skip a sub-subarray from (M+1) total copies that sums to (M+1)*S - target = S - R
        skip_sum = (M + 1) * S - target  # = S - R
        if skip_sum <= 0:
            # skip_sum can't be <= 0 unless R >= S (but R < S from remainder), or M+1=0 which can't happen.
            # So we won't get a valid skipping scenario here.
            candidate2 = math.inf
        else:
            # We want to skip a prefix or suffix with sum= skip_sum in up to 2 copies
            # Then the subarray length is (M+1)*n - that skip length
            # But skip sub-subarray must be either a minimal prefix or a minimal suffix to remain contiguous.
            prefix_len = min_prefix_sum_in_2(skip_sum)
            suffix_len = min_suffix_sum_in_2(skip_sum)
            
            best_skip = math.inf
            if prefix_len is not None:
                best_skip = min(best_skip, prefix_len)
            if suffix_len is not None:
                best_skip = min(best_skip, suffix_len)
            
            if best_skip == math.inf:
                candidate2 = math.inf
            else:
                candidate2 = (M + 1) * n - best_skip
        
        ans = min(candidate1, candidate2)
        return ans if ans != math.inf and ans > 0 else -1