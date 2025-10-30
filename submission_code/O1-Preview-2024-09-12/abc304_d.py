# YOUR CODE HERE
import sys
import threading
import bisect

def main():
    import sys
    import bisect
    import collections
    sys.setrecursionlimit(1 << 25)
    W, H = map(int, sys.stdin.readline().split())
    N = int(sys.stdin.readline())
    strawberries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    A = int(sys.stdin.readline())
    a_list = list(map(int, sys.stdin.readline().split()))
    B = int(sys.stdin.readline())
    b_list = list(map(int, sys.stdin.readline().split()))
    
    x_cuts = [0] + a_list + [W]
    x_cuts.sort()
    y_cuts = [0] + b_list + [H]
    y_cuts.sort()
    
    # Map strawberries to bins
    cell_counts = collections.Counter()
    for p, q in strawberries:
        x_bin = bisect.bisect_right(x_cuts, p) - 1
        y_bin = bisect.bisect_right(y_cuts, q) - 1
        cell_counts[(x_bin, y_bin)] += 1
    
    total_bins = (A+1)*(B+1)
    num_bins_with_strawberry = len(cell_counts)
    
    counts = cell_counts.values()
    max_count = max(counts)
    if num_bins_with_strawberry < total_bins:
        min_count = 0
    else:
        min_count = min(counts)
    print(f"{min_count} {max_count}")

threading.Thread(target=main).start()