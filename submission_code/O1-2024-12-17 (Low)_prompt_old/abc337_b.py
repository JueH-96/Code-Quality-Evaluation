def solve():
    import sys
    
    S = sys.stdin.readline().strip()
    
    # We will try to split S into three parts:
    # 1) All A's (possibly zero length)
    # 2) All B's (possibly zero length)
    # 3) All C's (possibly zero length)
    #
    # A direct approach:
    # 1. Count how many consecutive 'A's from the start.
    # 2. Then count how many consecutive 'B's from that point.
    # 3. Then count how many consecutive 'C's from that point.
    # 4. If we exhaust the string exactly after these parts, it is valid.
    
    n = len(S)
    
    # Count leading 'A's
    i = 0
    while i < n and S[i] == 'A':
        i += 1
    
    # Count next 'B's
    j = i
    while j < n and S[j] == 'B':
        j += 1
    
    # Count next 'C's
    k = j
    while k < n and S[k] == 'C':
        k += 1
    
    # Check if we've used up the entire string
    if k == n:
        # We also need to ensure that parts identified are truly single-type segments:
        # By construction, the first segment is all A's, second is all B's, and third is all C's.
        # If we consumed the entire string accordingly, it's valid.
        print("Yes")
    else:
        print("No")

# Call solve() after defining it
solve()