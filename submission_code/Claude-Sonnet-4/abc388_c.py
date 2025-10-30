import bisect

n = int(input())
a = list(map(int, input().split()))

count = 0
for i in range(n):
    # For mochi i as bottom, find how many can be placed on top
    max_top_size = a[i] // 2  # Integer division since we need a <= b/2
    
    # Find the rightmost position where a[j] <= max_top_size
    # using binary search
    pos = bisect.bisect_right(a, max_top_size)
    
    # All mochi from index 0 to pos-1 can be placed on top of mochi i
    # But we need to exclude mochi i itself if it's in that range
    valid_tops = pos
    if i < pos:
        valid_tops -= 1
    
    count += valid_tops

print(count)