import collections

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        left = 0
        cost = 0

        # mono_q stores (value, index) in decreasing order of value
        # to find the max in the window [left, right] efficiently.
        mono_q = collections.deque()
        
        # terms stores (value, count) to calculate part of the cost adjustment
        # when `left` is incremented. It represents the sequence
        # max(nums[left..l]) for l from left to right.
        terms = collections.deque()
        terms_sum = 0
        
        for right in range(n):
            # 1. Update `terms` and `terms_sum` for the new element nums[right].
            # This structure helps compute the change in cost when `left` increases.
            # terms_sum for a window [i, j] calculates sum_{l=i..j} max(nums[i..l]).
            count = 1
            val = nums[right]
            while terms and terms[-1][0] < val:
                p_val, p_count = terms.pop()
                terms_sum -= p_val * p_count
                count += p_count
            terms.append((val, count))
            terms_sum += val * count
            
            # 2. Update `cost` by extending the window to `right`.
            # The cost change is max(nums[left..right]) - nums[right].
            while mono_q and mono_q[-1][0] < nums[right]:
                mono_q.pop()
            mono_q.append((nums[right], right))
            cost += mono_q[0][0] - nums[right]
            
            # 3. Shrink window from the left if cost exceeds k.
            while cost > k:
                # The element nums[left] is being removed from the window.
                
                # Update mono_q: if the maximum element is at `left`, remove it.
                if mono_q and mono_q[0][1] == left:
                    mono_q.popleft()
                
                # Update cost:
                # cost(left+1, right) = cost(left, right) - cost_reduction
                # cost_reduction = (max(nums[left..right]) - nums[left]) +
                #                  sum_{l=left+1..right} (max(nums[left..l]) - max(nums[left+1..l]))
                # The sum term is sum_{l=left+1..right} max(0, nums[left] - max(nums[left+1..l])).
                
                # `terms_sum` currently stores sum_{l=left..right} max(nums[left..l]).
                # We need to find sum_{l=left+1..right} max(nums[left+1..l]).
                # Let's calculate cost reduction due to removing nums[left].
                
                terms_sum -= terms[0][0] * terms[0][1]
                if terms[0][1] > 1:
                    terms[0] = (terms[0][0], terms[0][1] - 1)
                    terms_sum += terms[0][0] * terms[0][1]
                else: # count is 1
                    terms.popleft()

                cost_reduction = mono_q[0][0] * (right - left) - terms_sum
                
                cost -= cost_reduction
                left += 1

            # All subarrays nums[i..right] where i is in [left, right] are valid.
            ans += (right - left + 1)
            
        return ans