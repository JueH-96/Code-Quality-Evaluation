def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    x_max = float('inf')
    has_positive_a = False
    for i in range(n):
        if a[i] > 0:
            has_positive_a = True
            x_max = min(x_max, q[i] // a[i])
    if not has_positive_a:
        x_max = sum(q) + 1 # Effectively infinity, set to a large enough value
        
    y_max = float('inf')
    has_positive_b = False
    for i in range(n):
        if b[i] > 0:
            has_positive_b = True
            y_max = min(y_max, q[i] // b[i])
    if not has_positive_b:
        y_max = sum(q) + 1 # Effectively infinity
        
    max_servings = 0
    
    if x_max <= y_max:
        for x in range(max(0, min(x_max + 1, 1000001))):
            current_y_max = float('inf')
            found_positive_b = False
            for i in range(n):
                if b[i] > 0:
                    found_positive_b = True
                    current_y_max = min(current_y_max, (q[i] - x * a[i]) // b[i])
            if not found_positive_b:
                current_y_max = y_max
                
            current_y = max(0, int(current_y_max))
            if current_y < 0:
                current_y = 0
            max_servings = max(max_servings, x + current_y)
    else:
        for y in range(max(0, min(y_max + 1, 1000001))):
            current_x_max = float('inf')
            found_positive_a = False
            for i in range(n):
                if a[i] > 0:
                    found_positive_a = True
                    current_x_max = min(current_x_max, (q[i] - y * b[i]) // a[i])
            if not found_positive_a:
                current_x_max = x_max
                
            current_x = max(0, int(current_x_max))
            if current_x < 0:
                current_x = 0
            max_servings = max(max_servings, current_x + y)
            
    print(max_servings)

if __name__ == '__main__':
    solve()