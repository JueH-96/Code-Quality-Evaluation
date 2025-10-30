import collections
from typing import List

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # current_P_sum: Stores the sum of adjusted values (P_k) for elements
        # in the current window [left_ptr, right_idx].
        current_P_sum = 0
        
        # current_nums_sum: Stores the sum of original values (nums[k]) for elements
        # in the current window [left_ptr, right_idx].
        current_nums_sum = 0
        
        left_ptr = 0
        ans = 0
        
        # dq: A deque to maintain the "monotonic P-value segments".
        # Each element in dq is a tuple (P_value, count).
        # P_value: The adjusted value for a segment of elements in the current window.
        # count: The number of elements in that segment that share this P_value.
        # The P_value's in dq are strictly increasing. This structure allows efficient
        # calculation of current_P_sum and its updates.
        dq = collections.deque() 

        for right_idx in range(n):
            val = nums[right_idx]
            
            # Add current element's original value to the sum.
            current_nums_sum += val

            # Determine the P_value for `nums[right_idx]` based on `P_{right_idx-1}`.
            # `P_value` for `nums[right_idx]` is `max(nums[right_idx], P_{right_idx-1})`.
            # `P_{right_idx-1}` is effectively `dq[-1][0]` if `dq` is not empty.
            
            # `P_val_for_right_idx_base` is the P_value of the segment immediately preceding `nums[right_idx]`.
            P_val_for_right_idx_base = 0
            if dq:
                P_val_for_right_idx_base = dq[-1][0]
            
            # `P_val_for_right_idx` is the final adjusted value for `nums[right_idx]`.
            P_val_for_right_idx = max(val, P_val_for_right_idx_base)

            # `num_elements_in_current_segment` accounts for `nums[right_idx]` and any
            # elements to its left that it "absorbs" due to its `P_value_for_right_idx`.
            num_elements_in_current_segment = 1
            
            # Process `dq` from right to left (popping elements)
            # All segments in `dq` whose `P_value` is greater than or equal to `P_val_for_right_idx`
            # must be popped. Their elements will now be covered by `P_val_for_right_idx`.
            while dq and dq[-1][0] >= P_val_for_right_idx:
                prev_P_val, prev_count = dq.pop()
                current_P_sum -= prev_P_val * prev_count # Remove old contribution from current_P_sum
                num_elements_in_current_segment += prev_count # Add their count to the new segment
            
            # Add the new segment to `dq`.
            dq.append((P_val_for_right_idx, num_elements_in_current_segment))
            current_P_sum += P_val_for_right_idx * num_elements_in_current_segment

            # Calculate cost for the current window [left_ptr, right_idx].
            # Cost = sum(P_k - nums[k]). Since (P_k - nums[k]) >= 0, this is valid.
            # P_left_ptr - nums[left_ptr] is always 0. So sum starts from index left_ptr.
            cost_for_window = current_P_sum - current_nums_sum

            # Shrink window from left if cost exceeds k.
            while cost_for_window > k:
                # Remove `nums[left_ptr]` from `current_nums_sum`.
                current_nums_sum -= nums[left_ptr]

                # Remove contribution of `nums[left_ptr]` from `current_P_sum` using `dq`.
                # The leftmost element in the window corresponds to the first segment in `dq`.
                P_val_to_remove, count_to_remove = dq[0]
                
                # Subtract its P_value contribution for one element.
                current_P_sum -= P_val_to_remove

                if count_to_remove == 1:
                    # If this segment only covered one element, remove it entirely.
                    dq.popleft()
                else:
                    # Otherwise, just decrement the count of elements in the segment.
                    dq[0] = (P_val_to_remove, count_to_remove - 1)
                
                left_ptr += 1
                
                # Recalculate cost for the updated window.
                cost_for_window = current_P_sum - current_nums_sum 
            
            # Add the number of valid subarrays ending at `right_idx` to the total count.
            # All subarrays from `nums[left_ptr...right_idx]` to `nums[right_idx...right_idx]` are valid.
            ans += (right_idx - left_ptr + 1)
            
        return ans