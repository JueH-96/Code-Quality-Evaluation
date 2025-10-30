class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from bisect import bisect_left, bisect_right
        import sys
        import threading
        sys.setrecursionlimit(1 << 25)
        def main():
            n = len(coordinates)
            coords = coordinates
            # Coordinate compress the y_i's
            ys = [y for x, y in coords]
            ys_sorted = sorted(set(ys))
            y_to_idx = {y: idx for idx, y in enumerate(ys_sorted)}
            # Sort the coordinates by x increasing, y increasing
            sorted_coords = sorted([(x, y, idx) for idx, (x, y) in enumerate(coords)])
            # Initialize BIT
            size = len(ys_sorted)
            class BIT:
                def __init__(self, size):
                    self.tree = [(0, False)] * (size + 2)
                    self.size = size + 2
                def update(self, idx, val):
                    idx += 1
                    while idx < self.size:
                        if val[0] > self.tree[idx][0] or (val[0] == self.tree[idx][0] and val[1]):
                            self.tree[idx] = val
                        idx += idx & -idx
                def query(self, idx):
                    idx +=1
                    res = (0, False)
                    while idx > 0:
                        if self.tree[idx][0] > res[0]:
                            res = self.tree[idx]
                        elif self.tree[idx][0] == res[0] and self.tree[idx][1]:
                            res = self.tree[idx]
                        idx -= idx & -idx
                    return res
            bit = BIT(size)
            dp = [ (0, False) ] * n
            idx_in_sorted = {}
            for i, (_x, _y, idx) in enumerate(sorted_coords):
                idx_in_sorted[idx] = i
            for i in range(n):
                x_i, y_i, idx_i = sorted_coords[i]
                y_idx = y_to_idx[y_i]
                # Query the BIT up to y_i -1
                max_len, includes_k = bit.query(y_idx -1)
                if idx_i == k:
                    dp_i = (max_len +1, True)
                else:
                    dp_i = (max_len +1, includes_k)
                dp[idx_i] = dp_i
                # Update the BIT at y_i
                existing = bit.query(y_idx)
                if dp_i[0] > existing[0] or (dp_i[0] == existing[0] and dp_i[1]):
                    bit.update(y_idx, dp_i)
            # Now, find the maximum dp[i][0] where dp[i][1] == True
            max_length = 0
            for i in range(n):
                if dp[i][1]:
                    if dp[i][0] > max_length:
                        max_length = dp[i][0]
            print(max_length)
        threading.Thread(target=main).start()