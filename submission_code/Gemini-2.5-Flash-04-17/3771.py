class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        if k == 0:
            return True

        # Precompute the first and last occurrence index for each character
        first = {}
        last = {}
        for i, char in enumerate(s):
            if char not in first:
                first[char] = i
            last[char] = i

        # dp[i] = maximum number of disjoint special substrings that can be selected from the prefix s[0...i]
        dp = [0] * n

        # Iterate through the string from left to right
        for i in range(n):
            # Option 1: The maximum number of disjoint special substrings ending at or before index i is the same as ending at or before index i-1
            if i > 0:
                dp[i] = dp[i-1]

            # Option 2: Consider ending a special substring s[l...i] at index i
            # We iterate through possible start indices l from i down to 0
            current_min_first = n
            current_max_last = -1

            for l in range(i, -1, -1):
                char_l = s[l]
                current_min_first = min(current_min_first, first[char_l])
                current_max_last = max(current_max_last, last[char_l])

                # Condition 1 check: Any character 'c' present in s[l...i] must not appear outside [l, i] in the original string s.
                # This is equivalent to: min(first[c]) >= l AND max(last[c]) <= i for all c in s[l...i].
                # We are checking this condition as we iterate l downwards.
                # If current_max_last > i, it means there is at least one character in s[l...i] whose last occurrence is after index i.
                # In this case, s[l...i] cannot be a special substring ending at i.
                # Since current_max_last is non-decreasing as l decreases, if it's already > i, it will remain > i for smaller l.
                # So we can break the inner loop for this fixed i.
                if current_max_last > i:
                    break

                # If current_max_last == i, it means index i is the rightmost occurrence among characters in s[l...i].
                # Now we check the left boundary condition: min(first[c]) >= l for all c in s[l...i].
                # This is tracked by current_min_first.
                if current_max_last == i and current_min_first >= l:
                    # Condition 1 is fully satisfied for the substring s[l...i].

                    # Condition 2 check: The substring is not the entire string s.
                    # The substring s[l...i] has length i - l + 1. The entire string has length n.
                    # We need i - l + 1 < n, which is equivalent to (l, i) != (0, n-1).
                    is_not_entire_string = (l != 0 or i != n - 1)

                    if is_not_entire_string:
                        # Found a valid special substring s[l...i] ending at index i.
                        # If we select this substring, the previous disjoint special substrings must be entirely within s[0...l-1].
                        # The maximum number of such substrings is dp[l-1] (or 0 if l == 0).
                        prev_dp = dp[l - 1] if l > 0 else 0
                        dp[i] = max(dp[i], prev_dp + 1)

        # The maximum number of disjoint special substrings in the entire string s is dp[n-1].
        # We check if this maximum count is at least k.
        return dp[n - 1] >= k