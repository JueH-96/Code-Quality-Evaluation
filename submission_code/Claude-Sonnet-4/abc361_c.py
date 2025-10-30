n, k = map(int, input().split())
a = list(map(int, input().split()))

min_range = float('inf')

# More efficient approach: for each possible min value, find optimal solution
unique_vals = sorted(set(a))

for target_min in unique_vals:
    # We want target_min to be the minimum in final sequence
    # So we must keep at least one occurrence of target_min
    # And we can remove all elements < target_min
    
    # Count elements < target_min (these can be removed for free)
    smaller_count = sum(1 for x in a if x < target_min)
    
    # Remaining removals needed
    remaining_removals = k - smaller_count
    
    if remaining_removals < 0:
        continue
    
    # Now from elements >= target_min, we need to keep (n-k) total elements
    # and remove remaining_removals more elements
    candidates = [(a[i], i) for i in range(n) if a[i] >= target_min]
    
    if len(candidates) < n - k:
        continue
    
    # Sort by value to remove largest ones first
    candidates.sort()
    
    # Keep the first (n-k) elements
    keep_count = n - k
    kept_values = [candidates[i][0] for i in range(keep_count)]
    
    if kept_values:
        current_range = max(kept_values) - min(kept_values)
        min_range = min(min_range, current_range)

print(min_range)