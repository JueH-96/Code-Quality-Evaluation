from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        We want the number of subarrays whose bitwise AND equals k.

        OBSERVATION 1: Any subarray whose AND == k cannot include any "blocker" element
                       that is missing at least one bit required by k.  Formally, an element x
                       is a "blocker" if (x & k) != k.  (Meaning x does not have all bits that k has.)
                       Such an element causes the subarray AND to lose some bit(s) of k, so it
                       can never equal k.  Hence, the array naturally splits into segments
                       separated by these blockers.

        For each segment of length M (where no element is a blocker), we only consider sub-subarrays
        within that segment.  In that segment, each element has all bits of k.  The AND of any range
        in that segment will therefore have at least the bits of k.  To ensure the AND is *exactly* k,
        none of the subarray's elements can force an extra bit above k to remain in the AND.

        Define A[i] = nums[i] & (~k) for elements in the segment.  Any bit set in A[i] is a "bit above k".
        A sub-subarray in that segment has AND == k exactly if and only if the AND of the corresponding
        A[i]'s is 0.  (Because we require all those extra bits to be "knocked out" by at least one element
        having 0 in that bit position.)

        So, within each segment (where (nums[i] & k) == k), we build A[i] = nums[i] & ~k, and we want
        to count the number of sub-subarrays for which AND(A[l..r]) = 0.

        APPROACH FOR AND(A[l..r]) = 0:
          1. Let mask = OR of all A[i] in the segment.  If mask == 0, then all A[i] are 0, so
             every sub-subarray has AND = 0.  The count is M*(M+1)//2.
          2. Otherwise, let BSet be the set of bits present in "mask".  We need each such bit
             to appear as zero in at least one element of [l..r].  In other words, for each bit b in BSet,
             the subarray must contain at least one index i where that bit is cleared (A[i] has 0 in bit b).
          3. Collect pos[b] = all indices i where A[i] has bit b == 0.  If pos[b] is empty for some b,
             that means all A[i] have that bit = 1, so AND is never 0.  Answer = 0 for that segment.
          4. Otherwise, we can do a simple sweep from left to right:
               - Maintain last_not[b] = the largest index ≤ r where bit b is 0 in A[...] (i.e. from pos[b]).
               - For each r, we find minPos = min( last_not[b] for b in BSet ).
               - If minPos == -1, no valid sub-subarray ends at r (some bit never found a zero up to r).
               - If minPos >= 0, then any start l ∈ [0 .. minPos] yields a sub-subarray [l..r] containing
                 at least one zero for each b.  That gives (minPos + 1) sub-subarrays ending at r.
          5. Sum these (minPos + 1) values across r.

        Overall steps:
          - Split nums by blockers into segments.
          - For each segment, build the array A of that segment, then count sub-subarrays with AND=0
            as above, which corresponds to AND = k in the original array.
          - Sum across all segments.
        """

        n = len(nums)
        answer = 0

        def solve_segment(segment: List[int]) -> int:
            """
            Given a segment of elements, all of which satisfy (elem & k) == k,
            count how many sub-subarrays yield AND == k.
            Equivalently, let A[i] = segment[i] & ~k.  We count sub-subarrays whose A-lhs AND = 0.
            """
            M = len(segment)
            if M == 0:
                return 0

            # Build A = nums[i] & ~k
            A = [(x & ~k) for x in segment]

            # If all A[i] == 0 => every sub-subarray has AND=0 => count = M*(M+1)//2
            mask = 0
            for x in A:
                mask |= x
            if mask == 0:
                return M * (M + 1) // 2

            # Collect bits in mask
            bits = []
            for b in range(31):
                if (mask & (1 << b)) != 0:
                    bits.append(b)

            # pos[b] = sorted indices i where bit b of A[i] == 0
            pos = {}
            for b in bits:
                pos[b] = []
            for i in range(M):
                for b in bits:
                    if (A[i] & (1 << b)) == 0:
                        pos[b].append(i)

            # If for some bit b, pos[b] is empty => no sub-subarray can have that bit cleared => 0
            for b in bits:
                if not pos[b]:
                    return 0

            # We'll move r from 0..M-1, maintain last_not[b] for each b
            # last_not[b] = largest index ≤ r where A[index] has bit b = 0.
            # We find minPos = min of last_not[b], and the contribution is (minPos+1) if >=0.
            p = {b: 0 for b in bits}
            last_not = {b: -1 for b in bits}
            count = 0

            for r in range(M):
                for b in bits:
                    # Advance pointer p[b] while pos[b][p[b]] <= r
                    # so last_not[b] is the largest index i <= r
                    while p[b] < len(pos[b]) and pos[b][p[b]] <= r:
                        last_not[b] = pos[b][p[b]]
                        p[b] += 1

                # Now minPos = min of last_not[b] over all b
                minPos = min(last_not[b] for b in bits)
                if minPos >= 0:
                    count += (minPos + 1)

            return count

        # Split nums into segments separated by blockers
        start = 0
        for i in range(n):
            # A "blocker" is where (nums[i] & k) != k
            if (nums[i] & k) != k:
                # Solve the segment [start .. i-1]
                answer += solve_segment(nums[start:i])
                start = i + 1
        # Solve the final segment [start .. n-1]
        if start < n:
            answer += solve_segment(nums[start:n])

        return answer