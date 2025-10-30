from typing import List

class Solution:
  def separateSquares(self, squares: List[List[int]]) -> float:
    
    s_total = 0.0
    
    # Based on constraints:
    # indywidual l_i can be up to 10^9.
    # sum of l_i*l_i is up to 10^12.
    # This implies individual l_i must be <= sqrt(10^12) = 10^6.
    # If any l_i > 10^6, then l_i*l_i > 10^12, violating sum constraint.
    # So, max l_i is 10^6. Max l_i*l_i for one square is 10^12.
    # Max s_total is 10^12.
    
    # Store relevant square properties, converted to float for calculations.
    processed_squares = []
    for sq_item in squares:
        # sq_item is [x_i, y_i, l_i]
        y_i = float(sq_item[1])
        l_i = float(sq_item[2])
        
        area = l_i * l_i
        s_total += area
        
        processed_squares.append({
            'y': y_i, 
            'l': l_i, 
            'y_plus_l': y_i + l_i, # Top y-coordinate of the square
            'area': area          # Full area of the square
        })

    # Constraints: 1 <= squares.length, 1 <= l_i. So S_total > 0.
    # If S_total could be 0 (e.g. squares list empty or all l_i=0),
    # handling it would be: if s_total == 0: return 0.0
    # But here, s_total will be positive.

    target_area = s_total / 2.0

    def calculate_A_below(H_line: float) -> float:
        total_area_below = 0.0
        for sq_data in processed_squares:
            y_i = sq_data['y']
            l_i = sq_data['l']
            y_plus_l_i = sq_data['y_plus_l']
            
            if H_line <= y_i:
                # Line is at or below the square's bottom edge
                # No part of the square is below the line
                pass # area_below_for_this_square = 0.0
            elif H_line >= y_plus_l_i:
                # Line is at or above the square's top edge
                # Entire square is below the line
                total_area_below += sq_data['area']
            else: 
                # Line intersects the square: y_i < H_line < y_i + l_i
                # Part below is a rectangle of height (H_line - y_i) and width l_i
                total_area_below += l_i * (H_line - y_i)
        return total_area_below

    # Binary search for H
    # Search range for H:
    # Min possible y_i is 0. Max possible y_i + l_i is 10^9 + 10^9 = 2*10^9
    # (Even with l_i <= 10^6, y_i can be 10^9, so y_i + l_i can be ~10^9)
    low = 0.0
    high = 2.0 * (10**9) 
    
    # `ans` will store the current best H (smallest H found so far that satisfies condition)
    ans = high # Initialize ans with the upper bound of search range.

    # 100 iterations are standard for typical float precision requirements in competitive programming.
    # Range size is 2*10^9. After 100 iterations, interval size is (2*10^9) / 2^100.
    # 2^100 is approx 1.26 * 10^30. So interval size is approx 1.5 * 10^-21.
    # This is well within the 10^-5 tolerance.
    for _ in range(100): 
        mid = low + (high - low) / 2 # Using this form of mid calculation is robust for large low/high.
        current_A_below = calculate_A_below(mid)
        
        if current_A_below >= target_area:
            # mid is a potential answer, or it's too high.
            # We try to find a smaller H that still satisfies the condition.
            ans = mid
            high = mid
        else:
            # current_A_below < target_area
            # mid is too low. We need to increase H.
            low = mid
            
    return ans