class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        """
        We want to find the largest subset of nums that can be turned into the same value
        using at most k increments or decrements in total.

        Key idea: The most cost-effective way to make a subset of numbers all equal is
        to transform them all to their median (if allowed real numbers, any value between
        the two middle elements for an even-sized subset). The cost is the sum of absolute
        differences from that target. We need that total cost <= k.

        Since we only need the size of the largest such subset, we can do a classic
        "sliding window with medians" approach on the sorted array. We keep a window that
        we can afford to make all equal at some median. As we expand the window, if the cost
        of making all elements in the window equal exceeds k, we shrink from the left.

        Implementation detail:
          - Sort nums.
          - Use two heaps (max-heap for the left half, min-heap for the right half) to
            maintain a running median of the current window.
          - Keep track of the cost to transform the current window to the median.
          - Expand the window by adding one element at a time to the right.
          - If the cost is too large (> k), remove from the left until cost <= k again.

        This approach works in O(n log n) time overall for n up to 10^5, which is acceptable.
        """

        import heapq

        nums.sort()
        n = len(nums)

        # Two heaps: left (max-heap), right (min-heap)
        left, right = [], []
        # We'll store negative of the values in 'left' to simulate max-heap.
        
        # Lazy deletion dictionaries to handle removing specific values that have slid out
        # from the left of the window. (We do not remove them immediately, we mark them
        # and do actual popping when they appear at the top of a heap.)
        to_remove_left = {}
        to_remove_right = {}

        # Sums of the values in left and right heaps, for quick cost calculation
        sum_left, sum_right = 0, 0

        def get_median():
            # If heaps are balanced or left has one extra element, top of left is median
            # (when total length is odd).
            # If both heaps have same size, the median can be any value between
            # the tops of left and right, but we will just pick top of left for consistency.
            if len(left) > len(right):
                return -left[0]
            else:
                # same size, pick top of left
                return -left[0]

        def balance_heaps():
            # Ensure that the size property holds:
            # either len(left) == len(right) or len(left) == len(right)+1
            if len(left) < len(right):
                # move the smallest from right to left
                val = heapq.heappop(right)
                # Add negative to left
                sum_right_sub = val
                sum_left_add = val
                # update sums
                nonlocal sum_left, sum_right
                sum_right -= sum_right_sub
                sum_left += sum_left_add
                heapq.heappush(left, -val)
            elif len(left) > len(right) + 1:
                # move the largest from left to right
                val = -heapq.heappop(left)
                sum_left_sub = val
                sum_right_add = val
                sum_left -= sum_left_sub
                sum_right += sum_right_add
                heapq.heappush(right, val)

        def clean_top(heap, remove_dict, is_left_heap):
            # Clean out elements on top of heap that are marked for removal
            while heap and ((-heap[0]) if is_left_heap else heap[0]) in remove_dict and remove_dict[((-heap[0]) if is_left_heap else heap[0])]>0:
                top_val = (-heap[0]) if is_left_heap else heap[0]
                heapq.heappop(heap)
                remove_dict[top_val] -= 1
                if remove_dict[top_val] == 0:
                    del remove_dict[top_val]

        def add_number(x):
            # We add x to one of the heaps
            # If x <= current median, goes to left, else to right
            med = get_median() if left or right else x

            nonlocal sum_left, sum_right
            if x <= med:
                heapq.heappush(left, -x)
                sum_left += x
            else:
                heapq.heappush(right, x)
                sum_right += x
            balance_heaps()

        def remove_number(x):
            # We'll mark x for removal from whichever heap it belongs to
            # and adjust the sums accordingly.
            # We'll rely on lazy removal to actually pop it.
            med = get_median()

            nonlocal sum_left, sum_right
            if x <= med:
                # that means x must be in left
                sum_left -= x
                to_remove_left[x] = to_remove_left.get(x, 0) + 1
            else:
                sum_right -= x
                to_remove_right[x] = to_remove_right.get(x, 0) + 1
            # We'll attempt to rebalance if needed
            balance_heaps()

        def get_cost():
            # Cost = sum of |nums[i] - median| in the window
            # We can compute it knowing:
            #  - sum_left: sum of left half
            #  - sum_right: sum of right half
            #  - median
            #  - counts in each half
            med = get_median()
            left_count = len(left)
            right_count = len(right)
            # left elements: each is <= med in this structure,
            # but remember left stores negative
            # sum_left is the sum of actual values in left
            cost_left = med * left_count - sum_left
            # right elements: each is >= med
            cost_right = sum_right - med * right_count
            return cost_left + cost_right

        left_ptr = 0
        max_len = 0

        # We'll expand right pointer over sorted nums
        for right_ptr in range(n):
            # add nums[right_ptr]
            add_number(nums[right_ptr])

            # fix any lazy pops that are top of a heap
            clean_top(left, to_remove_left, True)
            clean_top(right, to_remove_right, False)

            # while cost > k, shrink
            while get_cost() > k and left_ptr <= right_ptr:
                remove_number(nums[left_ptr])
                left_ptr += 1
                clean_top(left, to_remove_left, True)
                clean_top(right, to_remove_right, False)

            # update max_len
            window_len = right_ptr - left_ptr + 1
            if window_len > max_len:
                max_len = window_len

        return max_len