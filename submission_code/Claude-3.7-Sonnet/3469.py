class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        max_height = 0
        for height in range(1, 101):  # Height capped at 100 due to constraints
            # Calculate balls needed for both color arrangements
            
            if height % 2 == 0:  # Even height
                red_odd = (height // 2) ** 2  # Red in odd rows (1,3,5...)
                blue_even = (height // 2) * (height // 2 + 1)  # Blue in even rows (2,4,6...)
                
                red_even = (height // 2) * (height // 2 + 1)  # Red in even rows
                blue_odd = (height // 2) ** 2  # Blue in odd rows
            else:  # Odd height
                red_odd = ((height + 1) // 2) ** 2
                blue_even = ((height - 1) // 2) * ((height - 1) // 2 + 1)
                
                red_even = ((height - 1) // 2) * ((height - 1) // 2 + 1)
                blue_odd = ((height + 1) // 2) ** 2
            
            # Check if either arrangement is possible
            case1 = red_odd <= red and blue_even <= blue  # Red in odd, blue in even
            case2 = red_even <= red and blue_odd <= blue  # Red in even, blue in odd
            
            if case1 or case2:
                max_height = height
            else:
                break  # If neither arrangement works, higher heights won't work either
        
        return max_height