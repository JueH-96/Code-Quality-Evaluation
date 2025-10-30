def can_transform(s, t, k):
    n, m = len(s), len(t)
    
    # If the difference in lengths is greater than k, it's impossible
    if abs(n - m) > k:
        return False
    
    # If total different characters plus length difference is greater than k, it's impossible
    min_ops = abs(n - m)  # minimum operations needed just for length adjustment
    i = j = 0
    while i < min(n, m):
        if s[i] != t[j]:
            min_ops += 1
        i += 1
        j += 1
    min_ops += max(n, m) - min(n, m)  # add remaining length difference
    
    if min_ops > k:
        return False
    
    return True

# Read input
k = int(input())
s = input().strip()
t = input().strip()

# Print result
print("Yes" if can_transform(s, t, k) else "No")