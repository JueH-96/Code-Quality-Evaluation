from typing import List
import sys

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Explanation:
        # We need to select k disjoint subarrays. Their combined contribution is
        #   sum_{i=1}^{k} [ (-1)^(i+1) * subarraySum[i] * (k - i + 1) ]
        # In other words, if we number the chosen subarrays 1..k (in left-to‐right order),
        # the i‐th subarray’s sum is multiplied by:
        #    C[i] = (-1)^(i+1) * (k - i + 1)
        # Hence, subarray 1 gets coefficient k, subarray 2 gets coefficient −(k−1), subarray 3 gets (k−2) and so on.
        # This problem reduces to choosing intervals (subarrays) in order (non overlapping) so that
        #   dp = sum_{i=1}^{k} C[i] * (sum of chosen subarray i)
        # We use dynamic programming that “peels off” one segment at a time.
        #
        # Let dp[j][i] be the maximum possible strength (for segments 1..j) when considering the first i+1 elements (indexed 0..i)
        # and having chosen exactly j segments.
        # We can optimize the transition by, for each segment j, doing a modified Kadane style scan.
        #
        # We will maintain an array "prev" representing the best total strength for having chosen j segments up to each index.
        # For j==0 (i.e. no segments chosen) we have value 0 everywhere.
        #
        # For each new segment (j from 1 to k) the new segment’s coefficient is:
        #    coeff = (-1)^(j+1) * (k - j + 1)
        # Now, while scanning through the array we want to “start” a new subarray segment at index i.
        # That new segment’s contribution if started at i is:
        #    candidate = coeff * nums[i] + (dp for j-1 using elements strictly before i).
        # We can take dp[j-1][i-1] (or 0 if i==0) when starting a new segment.
        # Then, we allow the segment to extend. This is analogous to Kadane’s algorithm – we maintain a best running value
        # “best_end” that is the best possible value we can obtain for the current segment ending exactly at the current index.
        #
        # Finally, we update dp[j][i] = max( dp[j][i-1], best_end ) so that at each index i, the best solution using j segments
        # up to index i is known.
        # We then update our “prev” (i.e. dp[j]) for the next iteration.
        
        INF = 10**18  # a very large number
        # For zero segments chosen, the strength is 0.
        prev_dp = [0] * n  
        
        # We iterate over segments: seg index 0 corresponds to the 1st chosen segment, etc.
        for seg in range(k):
            # coefficient for the (seg+1)-th segment:
            # For seg==0 (first segment): coefficient = (+1)*(k) = k.
            # For seg==1 (second segment): coefficient = (-1)*(k-1)
            # For seg==2 (third segment): coefficient = (+1)*(k-2), and so on.
            coeff = (1 if seg % 2 == 0 else -1) * (k - seg)
            dp = [-INF] * n  # dp for current segment selection
            best_end = -INF  # best sum for a segment ending at current index
            
            for i in range(n):
                # When starting a new segment at index i:
                # Use dp for previous segment j-1 from index (i-1) if available, or 0 for seg==0.
                prev_val = prev_dp[i-1] if i > 0 else 0
                start_candidate = coeff * nums[i] + prev_val
                # Either start new subarray at i, or extend the previous subarray (if beneficial).
                if i == 0:
                    best_end = start_candidate
                else:
                    best_end = max(start_candidate, best_end + coeff * nums[i])
                # dp[i] is the best we can do up to index i.
                dp[i] = best_end if i == 0 else max(dp[i-1], best_end)
            prev_dp = dp  # move onto the next segment
        
        return prev_dp[-1]



# The following code is for local testing and interactive use.
if __name__ == "__main__":
    import sys
    import sys
    data = sys.stdin.read().strip()
    if not data:
        # Run default testcases if none provided.
        tests = [
            ("[1,2,3,-1,2]", 3),
            ("[12,-2,-2,-2,-2]", 5),
            ("[-1,-2,-3]", 1)
        ]
        sol = Solution()
        for s,k in tests:
            # Convert string representation of list to a list of ints.
            nums = eval(s)
            print(sol.maximumStrength(nums, k))
    else:
        # Expected input format:
        # For example: 
        #   nums = [1,2,3,-1,2]
        #   k = 3
        # We parse the input accordingly.
        lines = data.splitlines()
        nums_line = ""
        k_line = ""
        for line in lines:
            line = line.strip()
            if line.startswith("nums"):
                nums_line = line
            elif line.startswith("k"):
                k_line = line
        if nums_line:
            # Extract list part from something like "nums = [1,2,3,-1,2]"
            nums_part = nums_line.split('=')[1].strip()
            nums = eval(nums_part)
        else:
            nums = []
        if k_line:
            k = int(k_line.split('=')[1].strip())
        else:
            # if k is not found, assume last token is k
            parts = data.split()
            k = int(parts[-1])
        sol = Solution()
        result = sol.maximumStrength(nums, k)
        sys.stdout.write(str(result))