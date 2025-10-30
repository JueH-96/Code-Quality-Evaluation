class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Helper to compute sum of odd-indexed rows (1,3,5,...) up to h
        # and sum of even-indexed rows (2,4,6,...) up to h.
        def sum_odd_even(h):
            # number of odd rows = ceil(h/2)
            odd_count = (h + 1) // 2
            sum_odd = odd_count * odd_count
            # number of even rows = floor(h/2)
            even_count = h // 2
            # sum of first even_count evens: 2 + 4 + ... + 2*even_count
            sum_even = even_count * (even_count + 1)
            return sum_odd, sum_even

        total = red + blue
        max_h = 0
        # maximum possible h satisfies h*(h+1)/2 <= total
        # we just try h from 1 upwards until that sum exceeds total
        running = 0
        h = 1
        while True:
            running += h
            if running > total:
                break
            sum_odd, sum_even = sum_odd_even(h)
            # Case 1: odd rows are red, even rows are blue
            if sum_odd <= red and sum_even <= blue:
                max_h = h
            # Case 2: odd rows are blue, even rows are red
            if sum_odd <= blue and sum_even <= red:
                max_h = h
            h += 1

        return max_h