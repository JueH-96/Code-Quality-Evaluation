class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        """
        We want the number of substrings whose count of '1's is >= (count of '0's)^2.
        Let:
           z = number of zeros in a substring
           o = number of ones in a substring
           L = length of that substring = z + o
        The condition is: o >= z^2  =>  (L - z) >= z^2  =>  L >= z^2 + z  =>  L >= z * (z + 1).

        Observing that s can be up to length 40,000:
        - A naive O(n^2) approach is too large (could be ~1.6 billion checks).
        - Notice if z > 200, then z^2 = 40,000+ which cannot be satisfied by any
          substring of length ≤ 40,000. Hence effectively z <= 200 in any valid substring.

        We can use prefix sums of zeros:
          prefixZeros[i] = number of zeros in s[:i]  (i.e. up to but not including index i).
        Then the substring s[l..r] has z = prefixZeros[r+1] - prefixZeros[l].

        We'll iterate r from 0..(n-1). Let c = prefixZeros[r+1].
        Then the number of zeros in s[l..r] is c - prefixZeros[l].
        If that equals z, we check if (r - l + 1) >= z*(z+1). Rearranging,
           l <= r + 1 - (z*(z+1)).
        We only need to check z up to min(c, 200), because z cannot exceed c
        (the total zeros up to r).

        We keep, for each possible zero-count "val", a sorted list of indices "i"
        where prefixZeros[i] == val. At step r, to find how many l's produce z zeros
        in s[l..r], we look for all l in occur[c - z], with l <= (r+1) - z*(z+1).
        We'll do a binary search to count how many such l's exist.

        Complexity:
         - We do n iterations for r.
         - For each r, we consider up to 201 values of z.
         - Each check uses a binary search O(log(n)).
         - Total ~ O(n * 201 * log(n)) which is around 40,000 * 201 * log(40,000) ~
           on the order of 100 million ops. In optimized Python this can be borderline,
           but typically can be done with efficient code.

        We'll implement this approach below.
        """

        import bisect

        n = len(s)
        # prefixZeros[i] = number of '0's in s[:i]
        prefixZeros = [0] * (n + 1)
        for i in range(n):
            prefixZeros[i+1] = prefixZeros[i] + (1 if s[i] == '0' else 0)

        # occur[val] will store a sorted list of indices i where prefixZeros[i] == val.
        from collections import defaultdict
        occur = defaultdict(list)
        # Initially, prefixZeros[0] = 0 at index 0.
        occur[0].append(0)

        ans = 0

        for r in range(n):
            # c = prefixZeros for s[:r+1]
            c = prefixZeros[r+1]
            # We'll try z from 0.. up to min(c, 200).
            # If z*(z+1) > (r+1), no need to check further because l must be >= 0.
            # Also if c-z < 0 then we can't have that scenario.
            upper_z = min(c, 200)
            for z in range(upper_z + 1):
                neededZerosPrefix = c - z
                if neededZerosPrefix < 0:
                    break
                lengthNeeded = z * (z + 1)
                Lz = (r + 1) - lengthNeeded  # l must be <= Lz
                if Lz < 0:
                    break
                # We want prefixZeros[l] = neededZerosPrefix, and l <= Lz.
                lst = occur[neededZerosPrefix]
                # Count how many indices are <= Lz via bisect_right.
                count_l = bisect.bisect_right(lst, Lz)
                ans += count_l

            # Now record prefixZeros[r+1] in occur
            occur[c].append(r+1)

        return ans