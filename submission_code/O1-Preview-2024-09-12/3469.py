class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        max_height = 0
        for starting_color in ['red', 'blue']:
            h = 0
            r_remaining = red
            b_remaining = blue
            row_num = 1
            while True:
                # Determine the color of the current row
                if starting_color == 'red':
                    is_red_row = (row_num % 2 == 1)
                else:
                    is_red_row = (row_num % 2 == 0)
                needed_balls = row_num
                if is_red_row:
                    # Use red balls
                    if r_remaining >= needed_balls:
                        r_remaining -= needed_balls
                    else:
                        break
                else:
                    # Use blue balls
                    if b_remaining >= needed_balls:
                        b_remaining -= needed_balls
                    else:
                        break
                h += 1
                row_num +=1
            if h > max_height:
                max_height = h
        return max_height