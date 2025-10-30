D = int(input())

def solve(D):
    ans = float('inf')
    # Try values of x up to sqrt(D)
    x = 0
    while x * x <= D:
        # For each x, find the best y
        # y^2 should be close to D - x^2
        target = D - x * x
        if target < 0:
            ans = min(ans, abs(x * x - D))
            break
            
        y = int(target ** 0.5)  # Try floor of sqrt
        
        # Check y and y+1
        for y_test in [y, y + 1]:
            if y_test >= 0:
                curr = abs(x * x + y_test * y_test - D)
                ans = min(ans, curr)
        
        x += 1
    
    return ans

print(solve(D))