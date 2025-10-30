class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        ans = 0
        total_balls = red + blue
        # Try all possible heights h until the total balls required exceeds available balls
        for h in range(1, total_balls + 1):
            total_needed = h * (h + 1) // 2
            if total_needed > total_balls:
                break
            # Number of odd-indexed rows and even-indexed rows
            odd_count = (h + 1) // 2
            even_count = h // 2
            # Sum of sizes on odd rows = 1 + 3 + 5 + ... = odd_count^2
            sum_odd = odd_count * odd_count
            # Sum of sizes on even rows = 2 + 4 + 6 + ... = even_count * (even_count + 1)
            sum_even = even_count * (even_count + 1)
            # Check both color-start options:
            # 1) Odd rows red, even rows blue
            # 2) Odd rows blue, even rows red
            if (sum_odd <= red and sum_even <= blue) or \
               (sum_odd <= blue and sum_even <= red):
                ans = h
        return ans