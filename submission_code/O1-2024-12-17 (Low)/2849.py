class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        """
        We define the "imbalance number" of a 0-indexed integer array arr of length n as
        the number of indices 0 <= i < n-1 in the sorted version of arr at which
        sarr[i+1] - sarr[i] > 1.

        We want to return the sum of the imbalance numbers over all subarrays of nums.

        Explanation of approach:

        1. Naive definition:
           For each subarray, we sort it and count how many "gaps of at least 2" (sarr[i+1]-sarr[i] >= 2)
           exist between consecutive distinct values in the subarray.
           Directly doing this sort for each subarray would be O(N^2 * N log N) for N ~ 1000, which is too large.

        2. More efficient counting:
           Observe that the imbalance of a subarray depends only on the set of distinct elements in the subarray
           and how they are spaced. Concretely:
             - If e1 < e2 < ... < e_k are the distinct elements in ascending order,
               then the imbalance is the number of i such that e_{i+1} - e_i >= 2.

           We can build subarrays incrementally in O(N^2) total steps by expanding the right boundary R
           for each fixed left boundary L, keeping track of:
             - The frequencies of each element in the current subarray.
             - A data structure that maintains the distinct elements in sorted order.
             - The current imbalance of that subarray.

           When we move R forward by 1, we either:
             - Increase the frequency of nums[R] if it's already in the subarray,
               or
             - Insert nums[R] into our "distinct set" structure if it was not present before.

           If we are inserting x into the sorted distinct set, we find where x fits among the existing distinct
           elements and update the imbalance count accordingly (because we only need to check the gap to
           its immediate predecessor and successor in the set).

        3. Implementation detail:
           - We'll use a frequency array freq[] (since 1 <= nums[i] <= N) to track how many times each number appears.
           - We'll maintain a sorted list of distinct elements.
           - We'll keep a running "big_gaps" count that is exactly the imbalance of the current subarray.
           - Upon adding a new distinct element x via bisect, we check its immediate neighbors in the sorted list
             to adjust big_gaps (subtract the old gap between neighbors, if any, and add new gaps with x).

           This approach is O(N^2 * N) in the worst case because inserting elements into a Python list can cost O(N).
           For N=1000, this can still be made to run within a reasonable time if written carefully.

        We'll implement that solution below.
        """

        from bisect import bisect_left, insort

        n = len(nums)
        answer = 0

        for start in range(n):
            # frequency array (1..n possible in problem constraints)
            freq = [0] * (n + 1)
            distinct_list = []
            big_gaps = 0  # current subarray's imbalance

            for end in range(start, n):
                x = nums[end]
                # if x is newly inserted:
                if freq[x] == 0:
                    # find position to insert x in distinct_list
                    pos = bisect_left(distinct_list, x)

                    # neighbor to the left
                    left_val = distinct_list[pos - 1] if pos - 1 >= 0 else None
                    # neighbor to the right
                    right_val = distinct_list[pos] if pos < len(distinct_list) else None

                    # If x will lie between left_val and right_val, then first remove
                    # the gap that was counted between left_val and right_val (if both exist)
                    if left_val is not None and right_val is not None:
                        # remove old gap if it was >= 2
                        if right_val - left_val >= 2:
                            big_gaps -= 1

                    # Now, add x to the list
                    insort(distinct_list, x)

                    # Add new gaps (left_val, x) and (x, right_val)
                    if left_val is not None:
                        if x - left_val >= 2:
                            big_gaps += 1
                    if right_val is not None:
                        if right_val - x >= 2:
                            big_gaps += 1

                freq[x] += 1

                # big_gaps holds the imbalance of this subarray
                answer += big_gaps

        return answer