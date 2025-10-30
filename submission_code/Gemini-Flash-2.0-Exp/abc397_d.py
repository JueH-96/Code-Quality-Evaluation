def solve():
    n = int(input())
    
    for x_plus_y in range(1, int(n**(1/3)) * 2 + 2):
        if n % x_plus_y == 0:
            x_squared_minus_xy_plus_y_squared = n // x_plus_y
            
            # Solve for x and y using x + y = x_plus_y and x^2 - xy + y^2 = x_squared_minus_xy_plus_y_squared
            # (x+y)^2 = x^2 + 2xy + y^2
            # x^2 - xy + y^2 = x_squared_minus_xy_plus_y_squared
            # 3xy = (x+y)^2 - (x^2 - xy + y^2)
            # xy = ((x+y)^2 - (x^2 - xy + y^2)) / 3
            
            if (x_plus_y**2 - x_squared_minus_xy_plus_y_squared) % 3 == 0:
                xy = (x_plus_y**2 - x_squared_minus_xy_plus_y_squared) // 3
                
                # x and y are roots of t^2 - (x+y)t + xy = 0
                # t = (x+y +- sqrt((x+y)^2 - 4xy)) / 2
                
                delta = x_plus_y**2 - 4 * xy
                if delta >= 0:
                    delta_sqrt = int(delta**0.5)
                    if delta_sqrt**2 == delta:
                        x = (x_plus_y + delta_sqrt) // 2
                        y = (x_plus_y - delta_sqrt) // 2
                        
                        if x > 0 and y > 0 and x**3 - y**3 == n:
                            print(x, y)
                            return
    
    print("-1")

solve()