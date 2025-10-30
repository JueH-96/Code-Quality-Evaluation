class Solution:
    def minimumOperations(self, num: str) -> int:
        """
        We want to delete as few digits as possible so that the remaining (in-order)
        number is divisible by 25. If we delete all digits, we get '0' which is
        divisible by 25.

        A number is divisible by 25 if its last two digits are 00, 25, 50, or 75.
        We'll attempt to form each of these patterns as the ending of the final number
        and compute how many deletions are needed. Additionally, we can consider just
        keeping one '0' (if any) or deleting everything (becomes '0').

        Approach in detail:
         1) The maximum cost is len(num) because we can remove all digits and end with '0'.
         2) If there is at least one '0', we can remove all other digits and keep just '0'
            with cost = len(num) - 1.
         3) Otherwise, try to form each pattern among ['00', '25', '50', '75'].
            - For 'xy' in ['00','25','50','75']:
              (a) Find from right to left the first occurrence of y. Let that index = pos1.
              (b) Then from pos1-1 down to 0 find x. Let that index = pos0.
              (c) If successful, the cost = the number of digits we must remove:
                      cost = (len(num)-1 - pos1) + (pos1 - 1 - pos0).
                  Explanation:
                    - (len(num)-1 - pos1) removes all digits after pos1,
                      so that digit becomes the last one.
                    - (pos1 - 1 - pos0) removes all digits strictly between
                      pos0 and pos1, so that num[pos0] is immediately before num[pos1].
            - Take the minimum cost across all patterns.
        4) Return the overall minimum.
        """

        n = len(num)
        # Best answer can never exceed n (remove all => final "0")
        min_cost = n  # Case of removing everything => "0" => divisible by 25

        # If we have at least one zero, we can keep just that '0' and remove others
        if '0' in num:
            min_cost = min(min_cost, n - 1)

        # Possible two-digit endings for a multiple of 25
        patterns = ["00", "25", "50", "75"]

        for pattern in patterns:
            p0, p1 = pattern[0], pattern[1]

            # Find rightmost occurrence of p1
            pos1 = -1
            for i in range(n - 1, -1, -1):
                if num[i] == p1:
                    pos1 = i
                    break
            if pos1 == -1:
                continue  # Can't form this pattern if we can't find the last digit

            # Find rightmost occurrence of p0 to the left of pos1
            pos0 = -1
            for j in range(pos1 - 1, -1, -1):
                if num[j] == p0:
                    pos0 = j
                    break
            if pos0 == -1:
                continue  # Can't form this pattern if we can't find the second-last digit

            # Compute cost = digits removed after pos1 + digits removed between pos0 and pos1
            # = (n-1 - pos1) + (pos1 - 1 - pos0)
            cost = (n - 1 - pos1) + (pos1 - 1 - pos0)
            min_cost = min(min_cost, cost)

        return min_cost