import sys
import bisect

def main():
    data = sys.stdin
    W, H = map(int, data.readline().split())
    N = int(data.readline())
    strawberries = [tuple(map(int, data.readline().split())) for _ in range(N)]
    A = int(data.readline())
    a = list(map(int, data.readline().split()))
    B = int(data.readline())
    b = list(map(int, data.readline().split()))

    # Number of horizontal segments
    B1 = B + 1

    # Count strawberries in each cell
    bucket_counts = {}
    for px, py in strawberries:
        ix = bisect.bisect_left(a, px)
        iy = bisect.bisect_left(b, py)
        key = ix * B1 + iy
        bucket_counts[key] = bucket_counts.get(key, 0) + 1

    total_cells = (A + 1) * B1
    used_cells = len(bucket_counts)

    # If there is any empty cell, the minimum is 0
    if used_cells < total_cells:
        mn = 0
    else:
        mn = min(bucket_counts.values())

    mx = max(bucket_counts.values())

    print(mn, mx)

main()