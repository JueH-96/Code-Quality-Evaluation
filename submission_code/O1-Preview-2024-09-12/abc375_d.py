# YOUR CODE HERE
import sys

def main():
    import sys
    import threading
    def solve():
        S = sys.stdin.readline().strip()
        from collections import defaultdict
        positions_dict = defaultdict(list)
        for idx, c in enumerate(S):
            positions_dict[c].append(idx)
        total_ans = 0
        for positions in positions_dict.values():
            positions.sort()
            m = len(positions)
            ans = 0
            for i in range(m - 1):
                gap = positions[i+1] - positions[i] -1
                ans += gap * (m - i -1)
            total_ans += ans
        print(total_ans)

    threading.Thread(target=solve).start()