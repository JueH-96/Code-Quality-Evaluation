from typing import List
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        """
        We want to split the indices of nums into groups so that:
          1) Each group contains indices all of the same value.
          2) The sizes of any two groups differ by at most 1.
        Our goal is to find the minimum number of such groups.

        ----------------------------------------------------------
        INSIGHT / CORE IDEA:

        Condition (2) implies all group sizes can only be k or k+1
        for some integer k.  In particular, if the largest group has
        size M, then every group must have size either M or M-1, or
        else the size-difference would exceed 1.

        Condition (1) says each group is "homogeneous" (contains only
        indices of the same value).  So for a value v that appears
        f = freq(v) times, all those f indices must go into one or
        more groups, each of which is size either M or M-1, but never
        mixing with other values.

        For such a grouping to be valid with "largest group size = M":
          • For each frequency f, the remainder f mod M must be either
            0 or (M-1); because the "last split" can only be size M-1
            (if it's smaller than that, say M-2, it would differ by 2
            from a full group of size M).
          • Hence for each f, f mod M ∈ {0, M-1}.
            - If f mod M = 0, we can split f into exactly f//M groups
              each of size M.
            - If f mod M = M-1, we split it into f//M groups of size M
              plus 1 group of size (M-1).

        Moreover, M must be a divisor of either f or (f+1).  Indeed:
          f mod M = 0  <=>  M divides f.
          f mod M = M-1 <=> M divides (f + 1).

        Therefore, for each distinct frequency f in nums, M must lie
        in (divisors of f) ∪ (divisors of f+1).  For the whole array,
        M must be in the intersection of those sets across all distinct
        frequencies.

        Once we have a candidate M in that intersection, the total
        number of needed groups is:

            groupCount(M) = Σ [ c_f * ( (f // M) + extra ) ]

        where:
          • c_f = how many distinct values appear exactly f times
          • extra = 1 if f mod M = M-1, else 0

        We compute this for all candidate M in the intersection and
        take the minimum.  That will be our answer.

        This method is both correct and efficient if we:
          1) Count how many distinct values share frequency f.
          2) For each distinct f, gather all divisors of f and (f+1).
          3) Intersect these sets across all f to find feasible M's.
          4) Compute groupCount(M) for each M in that intersection and
             pick the minimum.

        Time complexity considerations:
          • Each f ≤ n, and finding divisors is O(√f).
          • The intersection typically shrinks quickly.
          • This passes under typical constraints up to n = 10^5.

        ----------------------------------------------------------
        EXAMPLES:

        Example 1:
            nums = [3,2,3,2,3]
            freq(3)=3, freq(2)=2
            Distinct frequencies = {3,2}.
            A(3) = divisors(3) ∪ divisors(4) = {1,3} ∪ {1,2,4} = {1,2,3,4}
            A(2) = divisors(2) ∪ divisors(3) = {1,2} ∪ {1,3} = {1,2,3}
            Intersection I = {1,2,3,4} ∩ {1,2,3} = {1,2,3}.

            - M=1 => each index in its own group => total 5 groups
            - M=2 => leftover(3)=1 => group usage for freq=3 is 3//2+1=2,
                      leftover(2)=0 => usage=1, total=3 groups
            - M=3 => leftover(3)=0 => usage=1, leftover(2)=2 => that is (3-1),
                      usage=0+1=1, total=2 groups
            Minimum is 2 => matches example.

        Example 2:
            nums = [10,10,10,3,1,1]
            freq(10)=3, freq(3)=1, freq(1)=2
            Distinct frequencies = {3,1,2}.
            A(3) = {1,3} ∪ {1,2,4} = {1,2,3,4}
            A(1) = {1}   ∪ {1,2}   = {1,2}
            A(2) = {1,2} ∪ {1,3}   = {1,2,3}
            Intersection => {1,2,3,4} ∩ {1,2} ∩ {1,2,3} = {1,2}.

            - M=1 => total groups = sum of frequencies = 6
            - M=2 => leftover(3)=1 => group usage= (3//2 +1)=2 for freq=3,
                      leftover(1)=1 => usage= (1//2 +1)=1 for freq=1,
                      leftover(2)=0 => usage= (2//2)=1 for freq=2,
                      sum= 2+1+1=4
            Minimum is 4 => matches example.
        """

        # Count how many times each distinct value appears
        freq_map = Counter(nums)  # value -> frequency
        # Next, count how many distinct values share the same frequency
        # i.e. freq_count[f] = how many values appear exactly f times
        freq_count = Counter(freq_map.values())

        # Prepare a list of (frequency, count_of_that_frequency)
        freq_pairs = list(freq_count.items())  # [(f1, c1), (f2, c2), ...]

        # A function to find the divisors of x in O(√x)
        def get_divisors(x: int) -> set:
            s = set()
            d = 1
            while d * d <= x:
                if x % d == 0:
                    s.add(d)
                    s.add(x // d)
                d += 1
            return s

        # A(f) = divisors(f) union divisors(f+1)
        def divisors_union(f: int) -> set:
            return get_divisors(f).union(get_divisors(f + 1))

        # Build the intersection of A(f) over all distinct frequencies f
        # Start with the set for the first frequency, then intersect
        # with subsequent frequencies.
        all_fs = list(freq_count.keys())
        if not all_fs:
            # Edge case: if nums is empty (not per constraints though)
            return 0

        # Initialize I by A(f) of the first frequency
        initial_f = all_fs[0]
        I = divisors_union(initial_f)

        # Intersect with A(f) for each other frequency
        for f in all_fs[1:]:
            cand = divisors_union(f)
            I = I.intersection(cand)
            if not I:
                break  # Shouldn't really happen because M=1 is always in A(f)

        # Now I is the set of all M that (potentially) satisfy leftover in {0, M-1}
        # for every distinct frequency.  Next step: compute how many total groups
        # that M would produce, pick the minimum.

        # In principle I should never be empty, because M=1 divides everything/f+1
        # so there's always at least {1} in side. But let's do a safe check anyway:
        if not I:
            # fallback is that if everything failed, we can always use M=1,
            # meaning 1 item per group => n total groups.
            return len(nums)

        # Compute the total number of groups for each M in I, pick the minimum
        # groupCount(M) = sum_{(f,c_f)} [ c_f * ( f//M + (1 if f mod M = M-1 else 0)) ]
        # Because leftover = M-1 means we need one extra partial group.
        n = len(nums)
        best = n  # worst-case: each item is its own group => n
        for M in I:
            if M <= 0:  # just ignore if something odd got in
                continue
            curr_groups = 0
            # Sum usage across all frequencies
            for (f, count_f) in freq_pairs:
                # leftover = f mod M must be either 0 or M-1 (by definition of intersection)
                base = (f // M)
                leftover = (1 if (f % M) == (M - 1) else 0)
                curr_groups += count_f * (base + leftover)
            best = min(best, curr_groups)

        return best