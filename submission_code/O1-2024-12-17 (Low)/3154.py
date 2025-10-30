class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)

        # M[i] will store the maximum value of nums up to index i-1
        # so M[i] = max(nums[0], nums[1], ..., nums[i-1])
        M = [0] * (n+1)
        current_max = float('-inf')
        for i in range(n):
            M[i] = current_max
            current_max = max(current_max, nums[i])
        M[n] = current_max  # Not used directly, just for clarity

        # best_diff[i] will store the maximum value of M[j] - nums[j] for j < i
        best_diff = [float('-inf')] * (n+1)
        running_best_diff = float('-inf')
        for j in range(1, n+1):
            # Difference if we pick j-1 as the middle index (i.e., i < j-1)
            if M[j-1] != float('-inf'):  # Means there's a valid i < j-1
                diff = M[j-1] - nums[j-1]
                running_best_diff = max(running_best_diff, diff)
            best_diff[j] = running_best_diff

        # Now for each k in [2..n-1], compute best_diff[k] * nums[k]
        # best_diff[k] is max over j<k of (M[j] - nums[j]) ensuring i<j<k
        answer = 0
        for k in range(2, n):
            if best_diff[k] != float('-inf'):
                value = best_diff[k] * nums[k]
                answer = max(answer, value)

        return max(0, answer)