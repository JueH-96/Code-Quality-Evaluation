from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # Initialize the two arrays that will store the distributed elements.
        arr1 = []
        arr2 = []

        # According to the problem statement:
        # 1. The first element of nums (nums[0] in 0-based indexing, referred to as nums[1] in the 1-based description) is appended to arr1.
        arr1.append(nums[0])

        # 2. The second element of nums (nums[1] in 0-based indexing, referred to as nums[2] in the 1-based description) is appended to arr2.
        arr2.append(nums[1])

        # 3. For the remaining elements, starting from the third element (nums[2] in 0-based indexing),
        #    we apply a rule based on the last elements of arr1 and arr2.
        #    The loop iterates from the third element's index (2) up to the last element's index (len(nums) - 1).
        #    This corresponds to the i-th operation for i from 3 to n in the problem description, processing nums[i] (which is nums[i-1] in 0-based index).
        for i in range(2, len(nums)):
            # Get the last element currently in arr1 and arr2.
            # We use negative indexing (-1) to access the last element quickly.
            last_arr1 = arr1[-1]
            last_arr2 = arr2[-1]

            # The rule is: if the last element of arr1 is greater than the last element of arr2:
            if last_arr1 > last_arr2:
                # Append the current element from nums (nums[i]) to arr1.
                arr1.append(nums[i])
            else:
                # Otherwise (if the last element of arr1 is less than or equal to the last element of arr2),
                # append the current element nums[i] to arr2.
                # Since elements in nums are distinct, arr1[-1] cannot be equal to arr2[-1]
                # after the first two steps (as nums[0] != nums[1]).
                # Subsequent appends add new, distinct elements.
                # Therefore, the 'else' condition specifically covers
                # the case where last_arr1 < last_arr2.
                arr2.append(nums[i])

        # After the loop finishes, all elements from nums have been distributed.
        # The final result is the concatenation of arr1 and arr2.
        # The '+' operator for lists performs concatenation, creating a new list.
        result = arr1 + arr2

        # Return the final list.
        return result