from collections import defaultdict
from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        mid = n // 2
        left_part = s[:mid]
        right_part_reversed = s[mid:][::-1]  # Reverse the right part for easier comparison

        # Precompute prefix sums for left and reversed right parts
        def build_prefix(s_part):
            prefix = [defaultdict(int)]
            for i in range(len(s_part)):
                new_count = prefix[i].copy()
                new_count[s_part[i]] += 1
                prefix.append(new_count)
            return prefix

        left_prefix = build_prefix(left_part)
        right_prefix = build_prefix(right_part_reversed)

        answer = []
        for q in queries:
            a, b, c, d = q
            c -= mid
            d -= mid
            # Convert c, d to indices in right_part_reversed
            c_rev = mid - 1 - d
            d_rev = mid - 1 - c
            c, d = c_rev, d_rev

            # Check case 4: i not in left substring and j not in right substring
            # i ranges: [0, a-1] and [b+1, mid-1]
            # j ranges: [0, c-1] and [d+1, mid-1] in reversed right part
            valid = True

            # Check i in [0, a-1]
            start_i1, end_i1 = 0, a-1
            start_j1, end_j1 = 0, c-1
            if start_i1 <= end_i1 and start_j1 <= end_j1:
                for i in range(start_i1, end_i1 + 1):
                    j_in_right = i
                    if j_in_right < start_j1 or j_in_right > end_j1:
                        continue
                    if left_part[i] != right_part_reversed[j_in_right]:
                        valid = False
                        break
                if not valid:
                    answer.append(False)
                    continue

            # Check i in [b+1, mid-1]
            start_i2, end_i2 = b+1, mid-1
            start_j2, end_j2 = d+1, mid-1
            if start_i2 <= end_i2 and start_j2 <= end_j2:
                for i in range(start_i2, end_i2 + 1):
                    j_in_right = i
                    if j_in_right < start_j2 or j_in_right > end_j2:
                        continue
                    if left_part[i] != right_part_reversed[j_in_right]:
                        valid = False
                        break
                if not valid:
                    answer.append(False)
                    continue

            # Collect case2: i in [a, b], j not in [c, d]
            # j not in [c, d] => j < c or j > d
            left_required = defaultdict(int)
            # j ranges for case2 in right_part_reversed
            # j_min_case2 = a's j, j_max_case2 = b's j
            j_min_case2 = a
            j_max_case2 = b
            # Compute j in [0, c-1] and [d+1, mid-1]
            if j_min_case2 <= c-1:
                start_j, end_j = j_min_case2, min(j_max_case2, c-1)
                if start_j <= end_j:
                    counts = defaultdict(int)
                    for key in right_prefix[end_j + 1]:
                        counts[key] += right_prefix[end_j + 1][key]
                    for key in right_prefix[start_j]:
                        counts[key] -= right_prefix[start_j][key]
                    for key, val in counts.items():
                        left_required[key] += val
            if d+1 <= j_max_case2:
                start_j, end_j = max(j_min_case2, d+1), j_max_case2
                if start_j <= end_j:
                    counts = defaultdict(int)
                    for key in right_prefix[end_j + 1]:
                        counts[key] += right_prefix[end_j + 1][key]
                    for key in right_prefix[start_j]:
                        counts[key] -= right_prefix[start_j][key]
                    for key, val in counts.items():
                        left_required[key] += val

            # Collect case3: j in [c, d], i not in [a, b]
            right_required = defaultdict(int)
            # i ranges: j's i is in [c, d], i not in [a, b]
            # i in [c, d] in right_part_reversed corresponds to i in left_part not in [a, b]
            start_i_case3 = c
            end_i_case3 = d
            for j in range(start_i_case3, end_i_case3 + 1):
                i_in_left = j
                if a <= i_in_left <= b:
                    continue
                if i_in_left < 0 or i_in_left >= mid:
                    continue
                char = left_part[i_in_left]
                right_required[char] += 1

            # Get left and right substring counts
            # Left substring [a, b]
            left_sub = defaultdict(int)
            for key in left_prefix[b + 1]:
                left_sub[key] += left_prefix[b + 1][key]
            for key in left_prefix[a]:
                left_sub[key] -= left_prefix[a][key]

            # Right substring [c, d]
            right_sub = defaultdict(int)
            for key in right_prefix[d + 1]:
                right_sub[key] += right_prefix[d + 1][key]
            for key in right_prefix[c]:
                right_sub[key] -= right_prefix[c][key]

            # Check left_required can be satisfied by left_sub
            for key, cnt in left_required.items():
                if left_sub.get(key, 0) < cnt:
                    valid = False
                    break
            if not valid:
                answer.append(False)
                continue

            # Check right_required can be satisfied by right_sub
            for key, cnt in right_required.items():
                if right_sub.get(key, 0) < cnt:
                    valid = False
                    break
            if not valid:
                answer.append(False)
                continue

            # Subtract left_required from left_sub and right_required from right_sub
            for key, cnt in left_required.items():
                left_sub[key] -= cnt
                if left_sub[key] == 0:
                    del left_sub[key]

            for key, cnt in right_required.items():
                right_sub[key] -= cnt
                if right_sub[key] == 0:
                    del right_sub[key]

            # Check if remaining left_sub and right_sub are equal
            if left_sub != right_sub:
                valid = False

            answer.append(valid)

        return answer