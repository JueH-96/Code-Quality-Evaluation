class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        We want the number of contiguous subarrays of nums where the bitwise AND of
        all elements equals k.

        Key observations:
          1) If an element e is missing any bit that k has (i.e. (e & k) != k), then
             no subarray containing e can have AND = k. Such elements split the array
             into "segments" where each element has all bits of k.
          2) Within one such segment (where every element has all bits of k), the only
             remaining problem is extra bits (bits not in k). If all elements of a subarray
             share an extra bit, that bit will remain in the AND, making it larger than k.
             To force the AND down to k, each "extra" bit must be missing in at least one
             element of that subarray.

        Therefore, the plan is:
         - Split the array at "break" positions where (nums[i] & k) != k.
         - Within each valid segment (where all elements contain the bits of k),
           use the "cover" approach to ensure that every extra bit (bit not in k)
           is missing in at least one element of the subarray.
           
        We do this by a standard sliding-window “coverage” pattern:
         - Let extra_bits = all bit positions (0..30) not in k.
         - For each segment, we move a right pointer. For nums[right], we note which
           extra bits it is missing. We update counts in a structure that tells us
           how many elements in the current window are missing each extra bit. If
           every extra bit is missing in at least one element of the window, the
           window "covers" all extra bits, so its AND = k.
         - We then shrink from the left to find the minimal left that still covers
           all extra bits. The number of valid subarrays ending at right is then
           "left - segment_start + 1" (where segment_start is where the segment began).

        This yields the total count in O(n * number_of_extra_bits) time, which is
        acceptable for up to 10^5 length and ~30 possible bits.
        """

        # Identify which bits are "extra" (i.e., not set in k)
        # We'll only consider bits up to 31 because 0 <= nums[i], k <= 10^9
        extra_bits = []
        for bit in range(31):
            if not (k & (1 << bit)):  # bit not in k
                extra_bits.append(bit)
        num_extra = len(extra_bits)

        # Edge case: if there are no extra bits, then each subarray that doesn't
        # include a breaking element (where (e & k) != k) has AND == k automatically.
        # We just sum up the number of subarrays in valid segments.
        if num_extra == 0:
            # We only need to split on "breaking" elements
            answer = 0
            start = 0
            n = len(nums)
            for i in range(n):
                # Break if this element doesn't contain all bits of k
                if (nums[i] & k) != k:
                    # number of subarrays in [start..i-1]
                    length = i - start
                    answer += length * (length + 1) // 2
                    start = i + 1
            # final segment
            length = len(nums) - start
            answer += length * (length + 1) // 2
            return answer

        # Otherwise, we have to do the coverage logic for extra bits
        n = len(nums)
        answer = 0

        # countMissing[idx] = how many elements in [left..right] are
        # missing the bit extra_bits[idx].
        countMissing = [0] * num_extra
        covered = 0  # how many bits among extra_bits are "covered" (missing in at least one element in window)

        segment_start = 0
        left = 0

        for right in range(n):
            # If this element doesn't have all bits of k, we must break the segment
            if (nums[right] & k) != k:
                # Close out any leftover from the previous segment
                # number of valid subarrays from [segment_start..left-1] if left>segment_start
                # but we've already counted them on the fly, so we just reset state.
                for i in range(num_extra):
                    countMissing[i] = 0
                covered = 0
                left = right + 1
                segment_start = right + 1
                continue

            # Otherwise, update coverage for extra bits
            val = nums[right]
            # For each extra bit, check if this element is missing it
            for i, bit in enumerate(extra_bits):
                if not (val & (1 << bit)):  # missing that extra bit
                    if countMissing[i] == 0:
                        covered += 1
                    countMissing[i] += 1

            # Now try to shrink from the left if all extra bits are covered
            # "covered == num_extra" means we have each extra bit missing
            # at least once in [left..right].
            while covered == num_extra and left <= right:
                # All subarrays that end at 'right' and have start in [segment_start..left]
                # are valid. The minimal left that still covers is 'left'. So the
                # number of valid starts is (left - segment_start + 1).
                answer += (n - right)  # ← This would be for "starts at left", if we were counting subarrays that start at left...
                # But we want "how many subarrays end at right?" pattern or "all subarrays in [left..right..n-1]"?
                #
                # Actually, a simpler approach: we only need to count subarrays *within this segment*.
                # Standard coverage trick: once we have coverage at [left..right], then
                # any left' in [segment_start..left] also has coverage, because removing
                # prefix elements that do not break coverage still covers. So the minimal left
                # that covers is 'left'. The number of valid subarrays ending at right is
                # (left - segment_start + 1).
                #
                # We'll do that approach to accumulate the answer for each right.
                break_min_left = left
                answer += (break_min_left - segment_start + 1)

                # Next, we try to move left by 1 to see if coverage still holds
                val_left = nums[left]
                for i, bit in enumerate(extra_bits):
                    if not (val_left & (1 << bit)):  # it was missing that bit
                        countMissing[i] -= 1
                        if countMissing[i] == 0:
                            covered -= 1
                left += 1

        # We have counted all valid subarrays on the fly. Return the result.
        return answer