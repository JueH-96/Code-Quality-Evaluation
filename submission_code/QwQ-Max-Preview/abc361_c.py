import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    a = list(map(int, input[2:2+n]))
    m = n - k

    if m == 0:
        print(0)
        return

    sorted_a = sorted(a)
    sorted_unique = []
    prev = None
    for num in sorted_a:
        if num != prev:
            sorted_unique.append(num)
            prev = num

    min_diff = float('inf')

    for i in range(len(sorted_unique)):
        low = sorted_unique[i]
        left = i
        right = len(sorted_unique) - 1
        res_j = None
        while left <= right:
            mid = (left + right) // 2
            high = sorted_unique[mid]
            # Calculate count of elements >= low and <= high
            count = bisect.bisect_right(sorted_a, high) - bisect.bisect_left(sorted_a, low)
            if count >= m:
                res_j = mid
                right = mid - 1
            else:
                left = mid + 1
        if res_j is not None:
            current_diff = sorted_unique[res_j] - low
            if current_diff < min_diff:
                min_diff = current_diff

    print(min_diff)

if __name__ == "__main__":
    main()