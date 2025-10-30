from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        """
        Calculates the number of subarrays where the first and last elements are equal
        to the largest element in the subarray.
        
        This problem can be solved in O(N) time using a monotonic stack.
        The stack will store tuples of (value, count). `value` is a number from `nums`,
        and `count` is the number of valid subarrays ending with this `value` at its
        last appearance. The stack maintains values in a strictly decreasing order.
        """
        stack = []  # Stores (value, count) tuples
        total_count = 0
        
        for num in nums:
            # Pop elements from the stack that are smaller than the current number.
            # The current `num` acts as a "previous greater element" for any subsequent
            # numbers, invalidating the context for these smaller, popped numbers.
            while stack and stack[-1][0] < num:
                stack.pop()
            
            # This is the count of valid subarrays ending at the current `num`.
            # It's at least 1 (the subarray containing just `num`).
            current_num_count = 1
            
            # If the stack is not empty and the top element has the same value,
            # it means we are extending a sequence. The count on the stack (`stack[-1][1]`)
            # represents the number of valid starting points for subarrays ending with
            # the previous occurrence of this number. We can form that many new
            # subarrays ending at the current position.
            if stack and stack[-1][0] == num:
                # Add the count from the previous occurrences and pop the old entry.
                current_num_count += stack[-1][1]
                stack.pop()
            
            # Add the number of newly found valid subarrays to the total.
            total_count += current_num_count
            
            # Push the current number and its calculated count onto the stack.
            stack.append((num, current_num_count))
            
        return total_count