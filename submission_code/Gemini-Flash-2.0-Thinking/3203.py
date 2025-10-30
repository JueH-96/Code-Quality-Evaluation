class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        results = []

        for a, b, c, d in queries:
            possible = True
            for i in range(n // 2):
                j = n - 1 - i
                is_i_rearrangeable = a <= i <= b
                is_j_rearrangeable = c <= j <= d

                if not is_i_rearrangeable and not is_j_rearrangeable:
                    if s[i] != s[j]:
                        possible = False
                        break

            if not possible:
                results.append(False)
                continue

            mismatched_fixed_chars = {}
            for i in range(n // 2):
                j = n - 1 - i
                is_i_rearrangeable = a <= i <= b
                is_j_rearrangeable = c <= j <= d

                if not is_i_rearrangeable and not is_j_rearrangeable:
                    if s[i] != s[j]:
                        mismatched_fixed_chars[i] = j

            if not mismatched_fixed_chars:
                results.append(True)
                continue

            can_resolve = True
            s_first_rearrange = s[a : b + 1]
            s_second_rearrange = s[c : d + 1]

            from collections import Counter
            first_rearrange_counts = Counter(s_first_rearrange)
            second_rearrange_counts = Counter(s_second_rearrange)

            for i, j in mismatched_fixed_chars.items():
                char_i = s[i]
                char_j = s[j]

                if char_j in first_rearrange_counts and first_rearrange_counts[char_j] > 0:
                    first_rearrange_counts[char_j] -= 1
                else:
                    can_resolve = False
                    break

                if can_resolve and char_i in second_rearrange_counts and second_rearrange_counts[char_i] > 0:
                    second_rearrange_counts[char_i] -= 1
                elif can_resolve:
                    can_resolve = False
                    break

                if not can_resolve:
                    break

            results.append(can_resolve)

        return results