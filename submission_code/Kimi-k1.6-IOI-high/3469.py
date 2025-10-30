class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def compute_max_height(start_with_red: bool) -> int:
            red_used = 0
            blue_used = 0
            height = 0
            current_is_red = start_with_red
            current_row = 1
            
            while True:
                if current_is_red:
                    if red_used + current_row > red:
                        break
                    red_used += current_row
                else:
                    if blue_used + current_row > blue:
                        break
                    blue_used += current_row
                height += 1
                current_is_red = not current_is_red
                current_row += 1
            
            return height
        
        # Compute heights for both starting scenarios
        height_red_start = compute_max_height(True)
        height_blue_start = compute_max_height(False)
        
        return max(height_red_start, height_blue_start)