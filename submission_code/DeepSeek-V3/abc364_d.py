import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    a = list(map(int, data[idx:idx+N]))
    idx += N
    queries = []
    for _ in range(Q):
        b = int(data[idx])
        k = int(data[idx+1])
        queries.append((b, k))
        idx += 2
    a.sort()
    for b, k in queries:
        left = 0
        right = N - 1
        # Binary search to find the k-th closest
        # We need to find the k-th smallest |a_i - b|
        # To do this, we can find the position where |a_i - b| is the k-th smallest
        # We can use a sliding window approach or a binary search approach
        # Here, we use a binary search approach to find the k-th smallest distance
        # We need to find the k-th smallest element in the sorted list of |a_i - b|
        # To do this, we can precompute the distances and sort them, but that would be O(N log N) per query
        # Instead, we can use a binary search on the possible distances
        # The minimum possible distance is 0, and the maximum is max(|a[0] - b|, |a[-1] - b|)
        # We can perform a binary search on the distance and count how many a_i have |a_i - b| <= mid
        # If the count is >= k, we try a smaller distance, else we try a larger distance
        low = 0
        high = max(abs(a[0] - b), abs(a[-1] - b))
        result = 0
        while low <= high:
            mid = (low + high) // 2
            # Find the number of a_i such that |a_i - b| <= mid
            # This is equivalent to finding the number of a_i in [b - mid, b + mid]
            left_idx = bisect.bisect_left(a, b - mid)
            right_idx = bisect.bisect_right(a, b + mid)
            count = right_idx - left_idx
            if count >= k:
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        print(result)

if __name__ == "__main__":
    main()