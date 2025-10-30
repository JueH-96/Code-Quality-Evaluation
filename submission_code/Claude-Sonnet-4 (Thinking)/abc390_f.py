n = int(input())
A = list(map(int, input().split()))

total = 0
for L in range(n):
    distinct_set = set()
    segments = 0
    for R in range(L, n):
        if A[R] not in distinct_set:
            distinct_set.add(A[R])
            x = A[R]
            has_left = (x - 1) in distinct_set
            has_right = (x + 1) in distinct_set
            
            if has_left and has_right:
                # x fills a gap, merging two segments
                segments -= 1
            elif has_left or has_right:
                # x extends an existing segment
                pass
            else:
                # x creates a new isolated segment
                segments += 1
        
        total += segments

print(total)