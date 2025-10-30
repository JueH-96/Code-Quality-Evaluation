from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        """
        We want to ensure the final array has its median equal to k. The "median" here
        is defined as the element at index n//2 (0-based) after sorting, picking the
        larger one if there's a tie for even-length arrays.

        For that to happen in the final array:
          • At least (n//2 + 1) elements must be ≤ k  (so that index n//2 can be k)
          • At least (n - n//2) elements must be ≥ k
          • At least one element is exactly k

        We can freely reorder the array, and the cost is the sum of |final_value - original_value|
        over all elements.

        Strategy:
          1) Split nums into three lists:
             A = all values < k   (sorted ascending)
             B = all values = k
             C = all values > k   (sorted ascending)
             Let a, b, c be their sizes.

          2) We may transform exactly X of the smallest elements in C down to k
             and Y of the largest elements in A up to k. Each transformation cost
             is (val - k) for those taken from C, or (k - val) for those taken from A.

          3) After transforming X from C into k, and Y from A into k, we have:
               - a' = a - Y values still < k
               - b' = b + X + Y total values = k
               - c' = c - X values still > k

             Then:
               • # of elements ≤ k = a' + b' = (a - Y) + (b + X + Y) = a + b + X
                 We need a + b + X >= (n//2 + 1).
               • # of elements ≥ k = b' + c' = (b + X + Y) + (c - X) = b + c + Y
                 We need b + c + Y >= (n - n//2).
               • We also need at least one element to be exactly k, i.e. b' >= 1,
                 which requires (b + X + Y) >= 1.

          4) We define:
               L_needed = (n//2 + 1)
               R_needed = (n - n//2)

             Then from the inequalities:
               X >= L_needed - (a + b)
               Y >= R_needed - (b + c)
               b + X + Y >= 1

             Also 0 <= X <= c, 0 <= Y <= a.

          5) We compute the cost for any valid (X, Y). Cost is:
               cost = costC(X) + costA(Y)
             where
               costC(X) = sum of (C[i] - k) for the X smallest elements of C
                        = (sum of first X in sorted C) - X*k
               costA(Y) = sum of (k - A[i]) for the Y largest elements of A
                        = Y*k - (sum of largest Y in A)

             We try all feasible X in range, compute the minimal feasible Y, and track
             the minimal cost.

        This yields a correct and efficient O(n log n) solution (sorting + one pass).
        """

        import sys
        input_data = nums
        n = len(input_data)
        mid = n // 2  # The median index (0-based), picking the "larger" if even length

        # Partition into A (<k), B (=k), C (>k)
        A, B, C = [], [], []
        for v in input_data:
            if v < k:
                A.append(v)
            elif v > k:
                C.append(v)
            else:
                B.append(v)
        A.sort()
        C.sort()

        a = len(A)
        b = len(B)
        c = len(C)

        # Prefix sums for C (ascending). costC(X) = sum(C[0..X-1]) - X*k
        prefixC = [0] * (c + 1)
        for i in range(c):
            prefixC[i+1] = prefixC[i] + C[i]

        # Prefix sums for A (ascending). To raise Y largest from A, we sum A[a-Y..a-1].
        # costA(Y) = Y*k - (sum of those Y largest).
        prefixA = [0] * (a + 1)
        for i in range(a):
            prefixA[i+1] = prefixA[i] + A[i]

        def costC(X):
            if X == 0:
                return 0
            return prefixC[X] - X*k

        def costA(Y):
            if Y == 0:
                return 0
            # Sum of largest Y in A = prefixA[a] - prefixA[a-Y]
            return Y*k - (prefixA[a] - prefixA[a-Y])

        L_needed = mid + 1            # at least mid+1 <= k
        R_needed = n - mid            # at least (n-mid) >= k

        # Minimum X, Y from those inequalities
        # 1) a + b + X >= L_needed => X >= L_needed - (a + b)
        Xmin = max(0, L_needed - (a + b))
        # 2) b + c + Y >= R_needed => Y >= R_needed - (b + c)
        Ymin = max(0, R_needed - (b + c))
        # We also need b + X + Y >= 1 => if b>=1, that is automatically satisfied if X,Y>=0;
        # otherwise if b=0 => X+Y>=1.

        Xmax = c  # can't transform more from C than we have
        Ymax = a  # can't transform more from A than we have

        INF = 10**20
        ans = INF

        for X in range(Xmin, Xmax + 1):
            # If b>0, we don't need extra condition for "at least one k".
            # If b=0, then X+Y >=1 => Y >= 1 - X
            if b == 0:
                Y0 = max(Ymin, 1 - X)
            else:
                Y0 = Ymin
            if Y0 < 0:
                Y0 = 0  # can't be negative
            if Y0 > Ymax:
                continue  # not feasible
            # Compute cost
            curr = costC(X) + costA(Y0)
            if curr < ans:
                ans = curr

        return ans