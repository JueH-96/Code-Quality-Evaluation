n, k = map(int, input().split())
a = list(map(int, input().split()))

def check(target):
    # Try all starting positions
    for start in range(n):
        groups = 0
        curr = 0
        for i in range(n):
            curr += a[(start + i) % n]
            if curr >= target:
                groups += 1
                if groups == k:
                    return i == n - 1  # Must use exactly n pieces
                curr = 0
    return False

# Binary search for optimal value
lo, hi = 1, sum(a)
while lo < hi:
    mid = (lo + hi + 1) // 2
    if check(mid):
        lo = mid
    else:
        hi = mid - 1

optimal = lo

# Find all cuts used in any optimal division
used_cuts = set()

for start in range(n):
    # Try to create division starting from position 'start'
    assign = [-1] * n  # which person gets each piece
    person = 0
    curr = 0
    valid = True
    
    for i in range(n):
        idx = (start + i) % n
        assign[idx] = person
        curr += a[idx]
        
        if curr >= optimal and person < k - 1:
            person += 1
            curr = 0
        elif i == n - 1:
            # Check if this is a valid division
            if person != k - 1 or curr < optimal:
                valid = False
    
    if valid:
        # Find which cut lines are used in this division
        for i in range(n):
            next_i = (i + 1) % n
            if assign[i] != assign[next_i]:
                # Cut line i+1 is cut (1-indexed)
                used_cuts.add(i + 1)

print(optimal, n - len(used_cuts))