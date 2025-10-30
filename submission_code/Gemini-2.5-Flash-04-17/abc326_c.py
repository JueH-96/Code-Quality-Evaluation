import sys
import bisect

# Read input
# Use sys.stdin.readline for faster input reading
input = sys.stdin.readline
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the gift coordinates
# Sorting allows us to efficiently count gifts within a range.
A.sort()

max_gifts = 0

# Iterate through each gift coordinate A[i] as the potential start of the half-open interval [A[i], A[i]+M).
# It can be proven that an optimal interval always exists which starts exactly at one of the gift coordinates.
for i in range(N):
    # The current interval considered is [A[i], A[i]+M).
    # We need to count the number of gifts A[k] such that A[i] <= A[k] < A[i]+M.
    # Since the array A is sorted, all elements A[k] with k >= i satisfy A[k] >= A[i].
    # Thus, we only need to count elements A[k] with k >= i such that A[k] < A[i]+M.
    
    # The target upper bound for the interval is A[i] + M.
    target_upper_bound = A[i] + M
    
    # We use bisect_left to find the index of the first element in the sorted array A that is >= target_upper_bound.
    # bisect_left(A, x) returns the index p such that all elements A[0]...A[p-1] are less than x, and all elements A[p]...A[N-1] are greater than or equal to x.
    # This index p is the point where 'target_upper_bound' would be inserted to keep the list sorted.
    # All elements before index p are strictly less than target_upper_bound.
    p = bisect.bisect_left(A, target_upper_bound)
    
    # The gifts in the interval [A[i], A[i]+M) that originate from considering A[i] as the left endpoint
    # are those A[k] such that i <= k < N and A[k] < A[i]+M.
    # Based on the properties of the sorted array A and the index p from bisect_left,
    # the elements A[0], A[1], ..., A[p-1] are all less than A[i]+M.
    # Among these, the ones at index i or greater are A[i], A[i+1], ..., A[p-1].
    # These are exactly the gifts in the interval [A[i], A[i]+M) starting from A[i].
    # The number of such elements is the count of indices from i to p-1, inclusive.
    # This count is (p - 1) - i + 1 = p - i.
    
    current_gifts = p - i
    
    # Update the maximum number of gifts found so far
    max_gifts = max(max_gifts, current_gifts)

# The loop iterates N times. Inside the loop, bisect_left takes O(log N) time.
# Sorting takes O(N log N). Reading input takes O(N).
# Total time complexity is O(N log N).
# Space complexity is O(N) for storing the coordinates (or O(1) if sorting in-place is considered).

# Print the maximum number of gifts
print(max_gifts)