def solve():
    t = int(input())
    results = []
    
    for _ in range(t):
        s = input().strip()
        x = input().strip()
        y = input().strip()
        results.append(can_exist(s, x, y))
    
    for result in results:
        print(result)

def can_exist(s, x, y):
    x0, x1 = x.count('0'), x.count('1')
    y0, y1 = y.count('0'), y.count('1')
    
    # If X = Y, any T would satisfy the condition
    if x == y:
        return "Yes"
    
    # Calculate the potential length of T
    if y1 == x1:  # The equation turns into (X0 - Y0) * |S| = 0
        if x0 == y0:  # Both sides of the equation are 0
            # If both X and Y have the same number of 0s and 1s, but are different strings,
            # we need to check if T = "" (empty string) satisfies the condition
            t = ""
            fx = "".join(s if char == '0' else t for char in x)
            fy = "".join(s if char == '0' else t for char in y)
            return "Yes" if fx == fy else "No"
        else:  # Non-zero on the left, zero on the right - contradiction
            return "No"
    
    # We have (X0 - Y0) * |S| = (Y1 - X1) * |T|
    # |T| = (X0 - Y0) * |S| / (Y1 - X1)
    t_length_float = (x0 - y0) * len(s) / (y1 - x1)
    if t_length_float < 0 or t_length_float != int(t_length_float):
        return "No"  # |T| must be a non-negative integer
    
    t_length = int(t_length_float)
    
    # Special case: T = "" (empty string)
    if t_length == 0:
        fx = "".join(s if char == '0' else "" for char in x)
        fy = "".join(s if char == '0' else "" for char in y)
        return "Yes" if fx == fy else "No"
    
    # Determine a potential T
    t = ['?'] * t_length  # Initialize with placeholders
    
    # Calculate the length of the final strings
    final_length = x0 * len(s) + x1 * t_length
    
    # Compute the origins of each character in the final strings
    fx_origins = []  # List of tuples (is_from_s, index_in_origin)
    for char in x:
        if char == '0':
            for i in range(len(s)):
                fx_origins.append((True, i % len(s)))
        else:  # char == '1'
            for i in range(t_length):
                fx_origins.append((False, i % t_length))
    
    fy_origins = []  # List of tuples (is_from_s, index_in_origin)
    for char in y:
        if char == '0':
            for i in range(len(s)):
                fy_origins.append((True, i % len(s)))
        else:  # char == '1'
            for i in range(t_length):
                fy_origins.append((False, i % t_length))
    
    # Make sure both origins arrays have the same length
    if len(fx_origins) != len(fy_origins):
        return "No"
    
    # Check if the origins are compatible
    for i in range(len(fx_origins)):
        is_from_s_x, idx_x = fx_origins[i]
        is_from_s_y, idx_y = fy_origins[i]
        
        if is_from_s_x and is_from_s_y:
            # Both are from S, they should match
            if s[idx_x] != s[idx_y]:
                return "No"
        elif is_from_s_x and not is_from_s_y:
            # X's character is from S, Y's character is from T
            # Derive: t[idx_y] should be s[idx_x]
            if t[idx_y] == '?':
                t[idx_y] = s[idx_x]
            elif t[idx_y] != s[idx_x]:
                return "No"
        elif not is_from_s_x and is_from_s_y:
            # X's character is from T, Y's character is from S
            # Derive: t[idx_x] should be s[idx_y]
            if t[idx_x] == '?':
                t[idx_x] = s[idx_y]
            elif t[idx_x] != s[idx_y]:
                return "No"
        else:  # not is_from_s_x and not is_from_s_y
            # Both are from T, they should match
            # Let's check if t[idx_x] and t[idx_y] are compatible
            if t[idx_x] != '?' and t[idx_y] != '?' and t[idx_x] != t[idx_y]:
                return "No"
            if t[idx_x] != '?' and t[idx_y] == '?':
                t[idx_y] = t[idx_x]
            elif t[idx_x] == '?' and t[idx_y] != '?':
                t[idx_x] = t[idx_y]
    
    # If all constraints are satisfied, there exists a valid T
    return "Yes"

solve()