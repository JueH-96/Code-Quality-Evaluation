class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        # Initialize the result and a prefix sum list for the imbalance numbers
        result = 0
        imbalance_prefix_sum = [0]

        # Process all possible sliding windows (subarrays) from left to right
        for start in range(len(nums)):
            # Initialize the imbalance for the current window
            imbalance = 0
            # Initialize a list to track last seen positions of elements for the current window
            last_seen = [None] * (len(nums) + 1)

            # Process the elements from start to the right end of the array
            for end in range(start, len(nums)):
                # Update the last seen position of the current element
                last_seen[nums[end]] = end
                prev = last_seen[nums[end] - 1] if nums[end] > 1 else None
                next = last_seen[nums[end] + 1] if nums[end] < len(nums) else None

                # Calculate the impact on imbalance due to the current element
                if prev is None:
                    next_imbalance = next - end if next is not None else 0
                elif next is None:
                    next_imbalance = 0
                else:
                    next_imbalance = next - prev - 1

                if next is None:
                    prev_imbalance = end - prev if prev is not None else 0
                elif prev is None:
                    prev_imbalance = 0
                else:
                    prev_imbalance = next - prev - 1

                # Adjust the imbalance for the current window
                imbalance += next_imbalance
                imbalance -= prev_imbalance

                # Add the imbalance of the current window to the result
                result += imbalance

            # Update the prefix sum of imbalance numbers for the next window
            imbalance_prefix_sum.append(imbalance_prefix_sum[-1] - imbalance)

        # Add the total imbalance numbers from the prefix sum list
        result += sum(imbalance_prefix_sum)
        return result