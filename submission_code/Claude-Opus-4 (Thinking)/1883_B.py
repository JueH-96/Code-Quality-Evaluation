# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    
    # Count character frequencies
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    # Count odd frequencies
    odd_count = sum(1 for f in freq.values() if f % 2 == 1)
    
    remaining = n - k
    
    if remaining % 2 == 0:
        # Need all frequencies even
        if k < odd_count:
            print("NO")
        elif (k - odd_count) % 2 == 0:
            print("YES")
        else:
            print("NO")
    else:
        # Need exactly one frequency odd
        if odd_count == 0:
            print("YES" if k % 2 == 1 else "NO")
        elif odd_count == 1:
            print("YES" if k % 2 == 0 else "NO")
        else:
            # odd_count > 1
            if k < odd_count - 1:
                print("NO")
            elif (k - (odd_count - 1)) % 2 == 0:
                print("YES")
            else:
                print("NO")