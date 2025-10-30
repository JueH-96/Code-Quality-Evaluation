class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        max_h = 0
        total_balls = red + blue
        
        h = 1
        while True:
            # total balls needed for h rows
            needed = h * (h + 1) // 2
            if needed > total_balls:
                break
            
            # number of odd‐indexed rows and even‐indexed rows
            odd_count = (h + 1) // 2
            even_count = h // 2
            
            # sum of sizes for odd rows = 1 + 3 + 5 + ... = odd_count^2
            sum_odd = odd_count * odd_count
            # sum of sizes for even rows = 2 + 4 + ... + 2*even_count = even_count*(even_count+1)
            sum_even = even_count * (even_count + 1)
            
            # pattern A: odd rows are red, even rows are blue
            okA = (sum_odd <= red and sum_even <= blue)
            # pattern B: odd rows are blue, even rows are red
            okB = (sum_odd <= blue and sum_even <= red)
            
            if okA or okB:
                max_h = h
            
            h += 1
        
        return max_h