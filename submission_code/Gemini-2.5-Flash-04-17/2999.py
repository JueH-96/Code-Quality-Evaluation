class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # The allowed swap operation (i, j with j - i = 2) means we can swap
        # characters at indices (0, 2) and (1, 3).
        # This partitions the indices into two independent sets based on parity:
        # {0, 2} and {1, 3}. Characters can only be swapped within their set.
        # Therefore, s1 can be made equal to s2 if and only if:
        # 1. The multiset of characters at indices {0, 2} in s1 is equal to the multiset
        #    of characters at indices {0, 2} in s2.
        # 2. The multiset of characters at indices {1, 3} in s1 is equal to the multiset
        #    of characters at indices {1, 3} in s2.

        # Check condition 1: characters at even indices {0, 2}
        # Comparing sorted lists checks if the multisets are equal.
        even_indices_s1 = sorted([s1[0], s1[2]])
        even_indices_s2 = sorted([s2[0], s2[2]])

        # Check condition 2: characters at odd indices {1, 3}
        # Comparing sorted lists checks if the multisets are equal.
        odd_indices_s1 = sorted([s1[1], s1[3]])
        odd_indices_s2 = sorted([s2[1], s2[3]])

        # s1 can be made equal to s2 only if both conditions are met.
        return even_indices_s1 == even_indices_s2 and odd_indices_s1 == odd_indices_s2