import sys
import bisect

def main():
    import sys
    import bisect
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    meds = []
    for _ in range(N):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        meds.append((a, b))
    # Sort by a_i ascending
    meds.sort()
    # Compute prefix_sum
    prefix_sum = [0]
    for a, b in meds:
        prefix_sum.append(prefix_sum[-1] + b)
    total_sum = prefix_sum[-1]
    target = total_sum - K
    if target <= 0:
        print(1)
        return
    # Binary search for X in [1, meds[-1][0] + 1]
    left = 1
    right = meds[-1][0] + 1
    while left < right:
        mid = (left + right) // 2
        # Find number of medicines with a_i < mid
        m = bisect.bisect_left(meds, (mid,))
        sum_less_X = prefix_sum[m]
        if sum_less_X >= target:
            right = mid
        else:
            left = mid + 1
    print(left)

if __name__ == '__main__':
    main()