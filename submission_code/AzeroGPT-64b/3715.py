from bisect import bisect_left, bisect_right

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        segments = []
        prefix_sum, sum_of_overlaps = [0], [0]
        for l, r, c in coins:
            segments.append((l, 1, c))
            segments.append((r + 1, -1, c))
            prefix_sum.append(prefix_sum[-1] + c)

        segments.sort()
        active_segments = []
        max_sum = 0
        current_overlap, start_accumulation_index = 0, 1

        for i, (pos, delta, coin) in enumerate(segments):
            count_left = bisect_left(segments, (pos, -1)) - i
            count_right = i - bisect_right(segments, (pos, 1)) 
            if delta > 0:
                bisect_left(active_segments, (-pos, coin))
                overlap = count_left - count_right + current_overlap
                if overlap < k:
                    current_overlap = count_left + count_right - overlap + current_overlap
                    sum_of_overlaps.append(sum_of_overlaps[-1] + (k - overlap) * coin)
                else:
                    sum_of_overlaps.append(sum_of_overlaps[-1] + (k * coin))
            elif delta < 0:
                overlap = count_left - count_right + current_overlap - 1
                if overlap < k:
                    sum_of_overlaps.append(sum_of_overlaps[-1] + (k - overlap) * coin)
                else:
                    sum_of_overlaps.append(sum_of_overlaps[-1] + (k * coin))
                bisect_right(active_segments, (-pos, coin))

            max_sum = max(max_sum, (pos-start_accumulation_index) * prefix_sum[-1] - sum_of_overlaps[-1])
            start_accumulation_index = pos

        return max_sum