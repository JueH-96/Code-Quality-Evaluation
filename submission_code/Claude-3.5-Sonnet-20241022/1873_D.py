def solve():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input()
        
        # Count minimum operations needed
        ops = 0
        i = 0
        while i < n:
            # If we find a black cell
            if s[i] == 'B':
                # Make k cells white starting at this position
                ops += 1
                i += k
            else:
                i += 1
                
        print(ops)

solve()