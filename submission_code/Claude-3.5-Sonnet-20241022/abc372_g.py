def solve_case():
    n = int(input())
    a = []
    b = []
    c = []
    for _ in range(n):
        ai, bi, ci = map(int, input().split())
        a.append(ai)
        b.append(bi)
        c.append(ci)
    
    # Find max possible x value
    max_x = float('inf')
    for i in range(n):
        if a[i] > 0:
            max_x = min(max_x, (c[i] - 1) // a[i])
    
    if max_x == float('inf') or max_x <= 0:
        return 0
        
    ans = 0
    # For each x from 1 to max_x
    for x in range(1, max_x + 1):
        # Find max possible y value for this x
        max_y = float('inf')
        for i in range(n):
            if b[i] > 0:
                curr_max_y = (c[i] - a[i] * x - 1) // b[i]
                max_y = min(max_y, curr_max_y)
        
        if max_y > 0:
            ans += max_y
            
    return ans

def main():
    t = int(input())
    for _ in range(t):
        print(solve_case())

main()