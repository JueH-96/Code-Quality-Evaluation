import math

def solve():
    D = int(input())
    
    min_abs_diff = D 

    x = 0
    while True:
        x_squared = x * x
        
        if x_squared - D > min_abs_diff:
            break
            
        min_abs_diff = min(min_abs_diff, abs(x_squared - D))
        
        target_val_for_y_squared = D - x_squared
        
        if target_val_for_y_squared >= 0:
            y_candidate_floor = math.isqrt(target_val_for_y_squared)
            
            y_cand_floor_squared = y_candidate_floor * y_candidate_floor
            current_sum1 = x_squared + y_cand_floor_squared
            min_abs_diff = min(min_abs_diff, abs(current_sum1 - D))
            
            y_candidate_ceil = y_candidate_floor + 1
            y_cand_ceil_squared = y_candidate_ceil * y_candidate_ceil
            current_sum2 = x_squared + y_cand_ceil_squared
            min_abs_diff = min(min_abs_diff, abs(current_sum2 - D))
            
        if min_abs_diff == 0:
            break
            
        x += 1

    print(min_abs_diff)

solve()