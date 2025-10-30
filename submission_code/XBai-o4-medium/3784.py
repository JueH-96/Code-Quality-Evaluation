from typing import List
from collections import defaultdict

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        if n == 0 or k == 0:
            return [0] * n

        # Preprocess prefix counts
        prefix_counts = defaultdict(lambda: defaultdict(int))
        max_len = 0

        for word in words:
            current_len = len(word)
            max_len = max(max_len, current_len)
            for l in range(1, current_len + 1):
                p = word[:l]
                prefix_counts[l][p] += 1

        # Compute original_max and S
        original_max = defaultdict(int)
        S = defaultdict(set)

        for l in range(1, max_len + 1):
            current_counts = prefix_counts[l]
            if not current_counts:
                original_max[l] = 0
                S[l] = set()
                continue
            max_val = max(current_counts.values())
            original_max[l] = max_val
            for p in current_counts:
                if current_counts[p] == max_val:
                    S[l].add(p)

        # Find M
        M = 0
        for l in range(max_len, 0, -1):
            if original_max.get(l, 0) >= k:
                M = l
                break

        answer = [0] * n

        if M == 0:
            # No possible L
            for i in range(n):
                if n - 1 >= k:
                    answer[i] = 0
                else:
                    answer[i] = 0
            return answer

        # Now, process based on M
        if len(S[M]) == 1:
            p = next(iter(S[M]))
            bad_i = set()
            for i in range(n):
                if len(words[i]) >= M and words[i][:M] == p:
                    bad_i.add(i)

            # Build candidate_Ls excluding M
            candidate_Ls = []
            for l in range(max_len, 0, -1):
                if original_max.get(l, 0) >= k and l != M:
                    candidate_Ls.append(l)

            # Now, fill answer
            for i in range(n):
                if n - 1 < k:
                    answer[i] = 0
                    continue
                if i not in bad_i:
                    answer[i] = M
                else:
                    # Check if original_max[M] -1 >=k
                    if original_max[M] - 1 >= k:
                        answer[i] = M
                    else:
                        res = 0
                        for l in candidate_Ls:
                            if len(words[i]) < l:
                                adjusted_max = original_max[l]
                            else:
                                p_l = words[i][:l]
                                if p_l in S[l]:
                                    if len(S[l]) == 1:
                                        adjusted_max = original_max[l] - 1
                                    else:
                                        adjusted_max = original_max[l]
                                else:
                                    adjusted_max = original_max[l]
                            if adjusted_max >= k:
                                res = l
                                break
                        answer[i] = res
        else:
            # len(S[M]) > 1, so for all i, answer is M if possible
            for i in range(n):
                if n - 1 >= k:
                    answer[i] = M
                else:
                    answer[i] = 0

        return answer