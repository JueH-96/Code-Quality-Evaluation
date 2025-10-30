class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        # Compute mismatches between s[i] and s[n - 1 - i]
        mismatches = [0] * half
        for i in range(half):
            mismatches[i] = 1 if s[i] != s[n - 1 - i] else 0
        # Compute prefix sums of mismatches
        prefix_mismatches = [0] * (half + 1)
        for i in range(half):
            prefix_mismatches[i + 1] = prefix_mismatches[i] + mismatches[i]
        total_mismatches = prefix_mismatches[half]
        result = []
        for a, b, c, d in queries:
            # Left variable positions: [a, b]
            # Right variable positions mapped to left half: [start_i, end_i]
            start_i = max(0, n - 1 - d)
            end_i = min(half - 1, n - 1 - c)
            # Compute overlap between [a, b] and [start_i, end_i]
            overlap_start = max(a, start_i)
            overlap_end = min(b, end_i)
            sum_mismatches_a = prefix_mismatches[b + 1] - prefix_mismatches[a]
            sum_mismatches_c = prefix_mismatches[end_i + 1] - prefix_mismatches[start_i]
            if overlap_start <= overlap_end:
                sum_mismatches_overlap = prefix_mismatches[overlap_end + 1] - prefix_mismatches[overlap_start]
            else:
                sum_mismatches_overlap = 0
            mismatches_in_variable_positions = sum_mismatches_a + sum_mismatches_c - sum_mismatches_overlap
            total_mismatches_in_fixed_positions = total_mismatches - mismatches_in_variable_positions
            result.append(total_mismatches_in_fixed_positions == 0)
        return result