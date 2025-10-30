from bisect import bisect_left, insort
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        if idx == 0:
            return  # idx is 1-based, 0 is not a valid index
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def query_range(self, a, b):
        if a > b:
            return 0
        return self.query(b) - self.query(a - 1)

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        breaks = []
        # Initialize breaks
        for j in range(n):
            if colors[j] == colors[(j + 1) % n]:
                insort(breaks, j)
        max_len = n
        sum_l_plus_1 = FenwickTree(max_len)
        sum_count = FenwickTree(max_len)
        # Process initial breaks
        m = len(breaks)
        if m == 0:
            sum_l_plus_1.update(n, (n + 1))
            sum_count.update(n, 1)
        else:
            for i in range(m):
                a = breaks[i]
                b = breaks[(i + 1) % m]
                length = (b - a) % n
                if length == 0:
                    length = n
                sum_l_plus_1.update(length, (length + 1))
                sum_count.update(length, 1)
        res = []
        for q in queries:
            if q[0] == 1:
                k = q[1]
                sum1 = sum_l_plus_1.query_range(k, max_len)
                sum2 = sum_count.query_range(k, max_len)
                ans = sum1 - k * sum2
                res.append(ans)
            else:
                index = q[1]
                color_i = q[2]
                original_color = colors[index]
                if original_color == color_i:
                    continue
                # Collect affected breaks
                affected_js = [(index - 1) % n, index]
                to_remove = []
                for j in affected_js:
                    prev_j = j
                    next_j = (j + 1) % n
                    # Compute colors before update
                    prev_color = original_color if prev_j == index else colors[prev_j]
                    next_color = original_color if next_j == index else colors[next_j]
                    if prev_color == next_color:
                        to_remove.append(j)
                # Remove breaks
                for j in to_remove:
                    pos = bisect_left(breaks, j)
                    if pos < len(breaks) and breaks[pos] == j:
                        breaks.pop(pos)
                        # Find previous and next breaks
                        a = breaks[pos - 1] if pos > 0 else breaks[-1] if breaks else 0
                        b = breaks[pos] if pos < len(breaks) else breaks[0] if breaks else 0
                        # Original segments [a, j) and [j, b)
                        len1 = (j - a) % n
                        if len1 == 0:
                            len1 = n
                        len2 = (b - j) % n
                        if len2 == 0:
                            len2 = n
                        # Subtract original segments
                        sum_l_plus_1.update(len1, -(len1 + 1))
                        sum_count.update(len1, -1)
                        sum_l_plus_1.update(len2, -(len2 + 1))
                        sum_count.update(len2, -1)
                        # Add merged segment
                        new_len = (b - a) % n
                        if new_len == 0:
                            new_len = n
                        sum_l_plus_1.update(new_len, (new_len + 1))
                        sum_count.update(new_len, 1)
                # Update color
                colors[index] = color_i
                # Check for new breaks
                to_add = []
                for j in affected_js:
                    prev_j = j
                    next_j = (j + 1) % n
                    prev_color = color_i if prev_j == index else colors[prev_j]
                    next_color = color_i if next_j == index else colors[next_j]
                    if prev_color == next_color:
                        to_add.append(j)
                # Add new breaks
                for j in to_add:
                    pos = bisect_left(breaks, j)
                    if pos < len(breaks) and breaks[pos] == j:
                        continue
                    insort(breaks, j)
                    pos = bisect_left(breaks, j)
                    # Find previous and next breaks
                    a = breaks[pos - 1] if pos > 0 else breaks[-1] if breaks else 0
                    if len(breaks) > 1:
                        b = breaks[(pos + 1) % len(breaks)]
                    else:
                        b = breaks[0] if breaks else 0
                    # Original segment [a, b)
                    original_len = (b - a) % n
                    if original_len == 0:
                        original_len = n
                    # Subtract original segment
                    sum_l_plus_1.update(original_len, -(original_len + 1))
                    sum_count.update(original_len, -1)
                    # Add new segments [a, j) and [j, b)
                    len1 = (j - a) % n
                    if len1 == 0:
                        len1 = n
                    len2 = (b - j) % n
                    if len2 == 0:
                        len2 = n
                    sum_l_plus_1.update(len1, (len1 + 1))
                    sum_count.update(len1, 1)
                    sum_l_plus_1.update(len2, (len2 + 1))
                    sum_count.update(len2, 1)
        return res