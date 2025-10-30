class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # Explanation:
        # You are allowed one operation: pick a contiguous subarray S and an integer x,
        # and add x to every element in S.
        # The final frequency of k will be:
        #   (number of indices outside S that originally equal k) +
        #   (number of indices inside S that become k, i.e. those with nums[i] + x == k).
        #
        # Note that for an index i inside S to become k we require:
        #     nums[i] + x == k   ->   x = k - nums[i]
        # Since x is the same for every index in S, for any beneficial operation (x != 0)
        # you are forced to convert only one type of element.
        # In other words, if you want a number a (with a != k) to be turned into k,
        # you would choose x = k - a. Then, inside the chosen contiguous S, only indices
        # with value a will become k, while indices that were originally k would sadly
        # change to k + (k - a) (losing their k status).
        #
        # So if you choose a candidate a with a != k (and with x = k - a), the net change in
        # the frequency of k from S is:
        #     (# of indices in S that are a) - (# of indices in S that are k).
        #
        # Let baseline be the frequency of k outside any operation (i.e. in the original array).
        # Then for candidate a we want to maximize over all contiguous subarrays S the quantity:
        #         gain = (# of indices in S with value a) - (# of indices in S with value k)
        # and the final answer would be: baseline + max_{candidate a, S chosen}(gain).
        #
        # For the candidate a = k (i.e. x = 0) the operation does nothing; the final frequency would just
        # be baseline. So we want to decide whether we can boost the baseline by converting some a (a != k)
        # to k – and we must choose a contiguous subarray.
        #
        # We can compute the maximum contiguous subarray gain for each candidate a (a in [1, 50] but a != k)
        # by constructing an auxiliary array B of the same length where:
        #     B[i] = +1, if nums[i] == a  (good; convert a to k)
        #     B[i] = -1, if nums[i] == k  (bad; originally k but we’re forced to change them)
        #     B[i] = 0, otherwise (neutral)
        #
        # Then the best gain for candidate a is the maximum contiguous subarray sum in B.
        # We let best_gain be the largest such gain (taking 0 if no positive gain is possible).
        #
        # Finally the answer is:
        #     baseline + best_gain
        
        baseline = sum(1 for num in nums if num == k)
        best_gain = 0
        
        # Try converting each candidate value a (where a != k) to k by setting x = k - a.
        for a in range(1, 51):
            if a == k:
                continue  # x=0 does not change anything.
            
            cur_sum = 0
            candidate_best = 0
            for num in nums:
                if num == a:
                    cur_sum += 1    # This element becomes k.
                elif num == k:
                    cur_sum -= 1    # This element was originally k but will no longer be after the operation.
                # For any other number, 0 contribution.
                
                if cur_sum < 0:
                    cur_sum = 0  # Restart the subarray if the sum becomes negative.
                if cur_sum > candidate_best:
                    candidate_best = cur_sum
            
            if candidate_best > best_gain:
                best_gain = candidate_best
        
        return baseline + best_gain


# Sample testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: After adding -5 (i.e. x = 1 - 6 = -5) to subarray indices [2..5],
    # one possible final array is [1, 2, -2, -1, 0, 1] which has frequency 2 for k=1.
    print(sol.maxFrequency([1, 2, 3, 4, 5, 6], 1))  # Expected output: 2

    # Example 2: After adding 8 (i.e. x = 10 - 2 = 8) to subarray indices [1..9],
    # final array becomes [10, 10, 11, 12, 13, 13, 12, 11, 10, 10] which gives 4 instances of 10.
    print(sol.maxFrequency([10, 2, 3, 4, 5, 5, 4, 3, 2, 2], 10))  # Expected output: 4