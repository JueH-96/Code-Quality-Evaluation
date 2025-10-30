import bisect

N = int(input())
A = list(map(int, input().split()))

count = 0
for j in range(N):
    # For each mochi j as the bottom, find how many mochi can be placed on top
    # A mochi can be placed on top if its size <= A[j] // 2
    target = A[j] // 2
    
    # Find the rightmost position where a value <= target can be inserted
    idx = bisect.bisect_right(A, target)
    
    # All mochi from index 0 to idx-1 have size <= target
    count += idx

print(count)