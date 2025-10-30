import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Count existing (fixed) occurrences for each letter
        fixed_count = [0] * 26
        for ch in s:
            if ch != '?':
                fixed_count[ord(ch) - ord('a')] += 1

        # Number of '?' to fill
        m = s.count('?')
        if m == 0:
            return s

        # We'll choose x_c for each letter c to minimize
        #   sum_c [ fixed_count[c] * x_c + x_c*(x_c-1)/2 ]
        # Equivalently, greedily pick the next assignment with smallest marginal cost:
        # marginal(c, x) = fixed_count[c] + x

        # Build a min‐heap of (marginal, letter_index)
        heap = [(fixed_count[c], c) for c in range(26)]
        heapq.heapify(heap)

        # x[c] = how many '?' we assign to letter c
        x = [0] * 26

        # Do m assignments
        for _ in range(m):
            marginal, c = heapq.heappop(heap)
            x[c] += 1
            # Next marginal for this letter increases by 1
            heapq.heappush(heap, (marginal + 1, c))

        # Build a sorted list of the assigned letters (lex order: all 'a's, then 'b's, ...)
        assign = []
        for c in range(26):
            if x[c] > 0:
                assign.extend([chr(ord('a') + c)] * x[c])

        # Fill the '?' in s with these letters in order
        res = []
        ai = 0
        for ch in s:
            if ch == '?':
                res.append(assign[ai])
                ai += 1
            else:
                res.append(ch)

        return "".join(res)