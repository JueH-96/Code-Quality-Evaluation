n, m = map(int, input().split())
a = list(map(int, input().split()))

if sum(a) <= m:
    print("infinite")
else:
    a.sort()
    
    cumulative = 0
    ans = None
    for i in range(n):
        # For x = a[i], total subsidy = cumulative + a[i] * (n - i)
        # where cumulative = sum of a[j] for j < i
        total_at_ai = cumulative + a[i] * (n - i)
        
        if total_at_ai <= m:
            cumulative += a[i]
        else:
            # Answer is in range [a[i-1], a[i]) if i > 0, else [0, a[i])
            # For x in this range, total = cumulative + x * (n - i)
            # We want cumulative + x * (n - i) <= m
            # So x <= (m - cumulative) / (n - i)
            ans = (m - cumulative) // (n - i)
            break
    
    if ans is not None:
        print(ans)
    else:
        print("infinite")