class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        """
        The rows must alternate colours.  Therefore, for a triangle of height h:
           rows 1,3,5,… (odd indices) are one colour,
           rows 2,4,6,… (even indices) are the other colour.
        For any h we can try both possible starts (odd rows red / odd rows blue).

        Sum of first k odd numbers  = k²
        Sum of first k even numbers = k(k+1)

        We increase h until the total number of balls needed exceeds red+blue and
        keep the last feasible height.
        """
        total = red + blue
        answer = 0

        h = 1
        while h * (h + 1) // 2 <= total:          # total balls required for height h
            odd_rows  = (h + 1) // 2              # number of odd-indexed rows
            even_rows = h // 2                    # number of even-indexed rows

            odd_sum  = odd_rows * odd_rows        # balls in odd rows   (1+3+5+...)
            even_sum = even_rows * (even_rows + 1)  # balls in even rows (2+4+6+...)

            # Case 1: odd rows red, even rows blue
            if odd_sum <= red and even_sum <= blue:
                answer = h
            # Case 2: odd rows blue, even rows red
            elif odd_sum <= blue and even_sum <= red:
                answer = h

            h += 1

        return answer