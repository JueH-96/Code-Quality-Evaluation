class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The problem asks to minimize the length of the string by repeatedly
        # applying an operation. The operation involves picking an index i
        # and deleting the closest occurrence of s[i] to the left and right.

        # Let's consider a single character 'c' and its occurrences in the string.
        # If 'c' appears only once, no operation can remove it as there is no
        # closest left or right occurrence. Its count remains 1.

        # If 'c' appears twice, say at indices i and j (i < j).
        # Picking index i: no closest left 'c', closest right is at j. Delete s[j]. Count becomes 1.
        # Picking index j: closest left 'c' is at i, no closest right. Delete s[i]. Count becomes 1.
        # In either case, the count of 'c' can be reduced to 1.

        # If 'c' appears three or more times. Let the indices be i_1 < i_2 < ... < i_m, where m >= 3.
        # We can pick any index i_k where 1 < k < m. The closest occurrence of 'c'
        # to the left of i_k is at index i_{k-1} (the element just before i_k in the sorted indices list).
        # The closest occurrence of 'c' to the right of i_k is at index i_{k+1} (the element just after i_k).
        # The operation removes s[i_{k-1}] and s[i_{k+1}]. This reduces the count of 'c' by 2.
        # We can repeat this operation as long as the count of 'c' is >= 3.
        # A count N >= 3 can be reduced to N-2, then N-4, and so on, until the count is 1 or 2.
        # If the count becomes 2, we can perform one more operation (as described above)
        # to reduce the count to 1. If the count becomes 1, no more operations are possible for 'c'.

        # Thus, for any character 'c' that appears initially, its count can be reduced
        # to a minimum of 1. Characters that were not initially present cannot be created,
        # so they won't be in the final string.

        # The operations for one character 'c' might shift the indices of other characters
        # and even other occurrences of 'c'. However, the relative order of occurrences
        # of any specific character is preserved. The "closest" definition refers to
        # the indices in the *current* string. If the current indices of 'c' are
        # p_1 < p_2 < ... < p_m, picking p_k (where 1 < k < m) removes p_{k-1} and p_{k+1}.
        # This is always possible as long as m >= 3. The ability to reduce the count
        # of any character > 1 down to 1 holds true regardless of other characters or
        # index shifts.

        # The final minimized string will contain exactly one occurrence of each
        # unique character that was present in the original string.
        # The length of this string is simply the number of unique characters.

        # We can find the number of unique characters using a set.
        unique_chars = set(s)

        # The length of the minimized string is the number of unique characters.
        return len(unique_chars)