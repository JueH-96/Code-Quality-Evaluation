from typing import List
import math

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)

        # Function to check if a minimum score of x is possible
        def check(x):
            if x == 0:
                return True # 0 score is always possible with 0 moves

            # Calculate the minimum required visits for each index i
            # k[i] = ceil(x / points[i])
            k = [(x + points[i] - 1) // points[i] for i in range(n)]

            # The path starts at -1, makes the first move to 0.
            # This first move costs 1 move and counts as one visit to index 0.
            # Total m moves are made. Let the sequence of indices visited be p_1, ..., p_m,
            # where p_1 = 0 and p_t is the index landed on after t moves from -1.
            # We need count(i in p_1, ..., p_m) >= k[i] for all i.
            # The sequence p_1, ..., p_m is a path of length m starting at index 0 within [0, n-1].

            # Let k_prime[i] be the minimum required visits for index i,
            # considering the path starts at index 0 (p_1).
            # This is the same as k[i].

            # Calculate prefix sums of required visits
            S = [0] * n
            S[0] = k[0]
            for i in range(1, n):
                S[i] = S[i-1] + k[i]

            # Calculate suffix sums of required visits
            T = [0] * n
            T[n-1] = k[n-1]
            for i in range(n - 2, -1, -1):
                T[i] = T[i+1] + k[i]

            # Minimum moves required starting from index 0 to achieve visits V'_i >= k_i,
            # where V'_i is the count of index i in the path starting at 0.
            # This minimum number of moves (length of the path) is given by:
            # max(max_{0<=i<n} (sum_{j=0}^i k_j + i), max_{0<=i<n} (sum_{j=i}^n-1 k_j + n-1-i))
            # This is the minimum length of a path starting at 0 that visits index j at least k_j times.

            required_moves_from_0 = 0
            for i in range(n):
                # Min moves to cover prefix [0, i] starting at 0 and end at i
                # This requires visiting indices 0..i enough times.
                # Sum of required visits in [0, i] is S[i].
                # Distance from 0 to i is i.
                required_moves_from_0 = max(required_moves_from_0, S[i] + i)

                # Min moves to cover suffix [i, n-1] starting at 0 and end at i
                # This requires visiting indices i..n-1 enough times.
                # Sum of required visits in [i, n-1] is T[i].
                # Distance from 0 to i involves going to n-1 and coming back to i
                # Or going directly to i.
                # The formula max(S_i+i, T_i+n-1-i) covers both cases for ending at i.
                # The minimum moves from 0 to achieve V_j >= k_j over all j, regardless of ending point.
                # is max_{0<=i<n} (max( sum_{j=0}^i k_j + i, sum_{j=i}^n-1 k_j + n-1-i ))
                # The required moves from 0 to achieve counts k over [0, n-1] is this max value.

                # Let's use the formula directly for the minimum moves starting from 0
                required_moves_from_0 = max(required_moves_from_0, T[i] + (n - 1 - i))

            # The path p_1, ..., p_m starts at 0 and has length m.
            # The minimum length required for a path starting at 0 to achieve k visits is required_moves_from_0.
            # We have a path of length m. So we need m >= required_moves_from_0.
            return m >= required_moves_from_0


        # Binary search for the maximum possible minimum score
        low = 0
        # A loose upper bound for the maximum possible minimum score.
        # If we visit every point once, min score is min(points).
        # If we visit points proportional to m, it's roughly m/n * min(points).
        # Max score for one point is m * max(points).
        # Max possible minimum score should not exceed max(points).
        # Let's use max(points) as a potential part of the upper bound.
        # The answer x can be larger than max(points), e.g., points=[1, 100], m=100, x=100.
        # A safer upper bound: sum of points if visited once (sum(points)) * m? No.
        # Consider maximum possible score for any single point: m * max(points[i])
        # The minimum score will be limited by the least visited point.
        # Upper bound can be max(points) * m.
        # max(points) <= 10^6, m <= 10^9. Max bound around 10^15.
        # A tighter bound might be sum(points) or max(points) + m.
        # Example 1: max(points)=4. Output 4.
        # Example 2: max(points)=3. Output 2.
        # Let's try max(points) * n. 10^6 * 5 * 10^4 = 5 * 10^10.
        # Let's try a generously safe bound like 2 * 10^14.
        high = 2 * 10**14 
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return ans