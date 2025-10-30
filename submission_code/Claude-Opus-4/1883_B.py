# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    s = input().strip()
    
    # Count frequency of each character
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    # Remaining characters after removal
    remaining = n - k
    
    # Count how many characters have odd frequency
    odd_count = sum(1 for f in freq.values() if f % 2 == 1)
    
    # If remaining is odd, we can have at most 1 odd frequency
    # If remaining is even, we must have 0 odd frequencies
    
    if remaining % 2 == 1:
        # We need at most 1 odd frequency after removal
        # Currently we have odd_count odd frequencies
        # We need to reduce odd frequencies to at most 1
        
        # Each removal from an odd frequency makes it even
        # Each removal from an even frequency makes it odd
        # So we need to be careful about how we remove
        
        # The minimum number of odd frequencies we can achieve is:
        # - If k >= odd_count, we can make all odd frequencies even by removing 1 from each
        #   Then we have k - odd_count removals left
        #   These removals will create new odd frequencies
        #   So we'll have (k - odd_count) odd frequencies
        # - If k < odd_count, we can reduce odd_count by k, leaving odd_count - k
        
        min_odd = abs(odd_count - k)
        
        # Since remaining is odd, we need exactly 1 odd frequency
        # We can achieve this if min_odd <= 1
        if min_odd <= 1:
            print("YES")
        else:
            print("NO")
    else:
        # remaining is even, we need 0 odd frequencies
        # Similar logic as above
        min_odd = abs(odd_count - k)
        
        # We need exactly 0 odd frequencies
        # We can achieve this if min_odd == 0
        if min_odd == 0:
            print("YES")
        else:
            print("NO")

t = int(input())
for _ in range(t):
    solve()