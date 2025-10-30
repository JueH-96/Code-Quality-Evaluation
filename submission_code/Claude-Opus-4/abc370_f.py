def can_divide(A, K, target, start):
    """Check if we can divide the array starting from 'start' into K parts with each sum >= target"""
    n = len(A)
    parts = 0
    current_sum = 0
    
    for i in range(n):
        idx = (start + i) % n
        current_sum += A[idx]
        
        if current_sum >= target:
            parts += 1
            current_sum = 0
            if parts == K:
                return True
    
    return False

def find_cuts_used(A, K, target, start):
    """Find which cuts are used when dividing with given target starting from 'start'"""
    n = len(A)
    cuts_used = set()
    current_sum = 0
    parts = 0
    
    for i in range(n):
        idx = (start + i) % n
        current_sum += A[idx]
        
        if current_sum >= target:
            parts += 1
            if parts < K:  # Not the last part
                # Cut after position idx
                cuts_used.add(idx)
            current_sum = 0
    
    return cuts_used

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Binary search bounds
left, right = 1, sum(A)
max_min_sum = 0

# Try all starting positions
for start in range(N):
    # Binary search for this starting position
    l, r = left, right
    
    while l <= r:
        mid = (l + r) // 2
        if can_divide(A, K, mid, start):
            max_min_sum = max(max_min_sum, mid)
            l = mid + 1
        else:
            r = mid - 1

# Now find all cuts that are used in any optimal division
all_used_cuts = set()

for start in range(N):
    if can_divide(A, K, max_min_sum, start):
        cuts = find_cuts_used(A, K, max_min_sum, start)
        all_used_cuts.update(cuts)

# Number of cuts never used
never_cut = N - len(all_used_cuts)

print(max_min_sum, never_cut)