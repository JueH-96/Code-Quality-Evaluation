class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        t = len(ones)
        n = len(nums)
        
        candidates = set()
        # Add all original ones
        for x in ones:
            candidates.add(x)
            # Add adjacent positions
            if x > 0:
                candidates.add(x-1)
            if x < n-1:
                candidates.add(x+1)
        # Add ends
        candidates.add(0)
        candidates.add(n-1)
        # Add median if any
        if t > 0:
            median = ones[t//2]
            candidates.add(median)
            candidates.add(median-1)
            candidates.add(median+1)
        # Convert to sorted list
        candidates = sorted(candidates)
        
        min_total = float('inf')
        
        for x in candidates:
            initial_ones = nums[x]
            remaining = k - initial_ones
            if remaining <= 0:
                min_total = 0
                break  # best possible
            original_available = t - (1 if initial_ones else 0)
            max_m = min(original_available, remaining)
            min_m = max(0, remaining - maxChanges)
            if max_m < min_m:
                continue  # no feasible solution, but constraints say it's possible
            
            # Generate the list of distances from x to ones (excluding x)
            distances = []
            for pos in ones:
                if pos != x:
                    distances.append(abs(pos - x))
            # Sort distances
            distances.sort()
            m_max_possible = len(distances)
            # Create prefix sums for distances
            prefix = [0] * (m_max_possible + 1)
            for i in range(m_max_possible):
                prefix[i+1] = prefix[i] + distances[i]
            
            # Now compute for all m in [min_m, max_m]
            for m in range(min_m, max_m + 1):
                c = remaining - m
                if c < 0 or c > maxChanges:
                    continue
                # Sum original: first m elements in distances
                if m > m_max_possible:
                    continue  # not possible
                sum_original = prefix[m] if m <= m_max_possible else 0
                # Compute sum_created for c elements
                if c == 0:
                    sum_created = 0
                else:
                    # Compute sum of c minimal (1 + distance) terms around x
                    # Find the minimal sum of c terms of (d+1), sorted by d
                    # Use the earlier method to compute sum_created
                    sum_created = 0
                    # Function to compute sum_created given c and x
                    def compute_sum_created(x, c, n):
                        def s(D):
                            left = min(D, x)
                            right = min(D, n - x -1)
                            return left + right
                        # Find minimal D where s(D) >= c
                        low = 1
                        high = max(x, n - x -1) + 1
                        D = high  # initial value
                        while low <= high:
                            mid = (low + high) // 2
                            if s(mid) >= c:
                                D = mid
                                high = mid -1
                            else:
                                low = mid +1
                        # Now compute sum
                        sum_left = 0
                        a = min(D-1, x)
                        sum_left = a * (a + 3) // 2
                        sum_right = 0
                        b = min(D-1, n - x -1)
                        sum_right = b * (b + 3) // 2
                        sum_total = sum_left + sum_right
                        sum_s = s(D-1)
                        remaining_terms = c - sum_s
                        sum_total += remaining_terms * (D +1)
                        return sum_total
                    sum_created = compute_sum_created(x, c, n)
                total = sum_original + sum_created
                if total < min_total:
                    min_total = total
        
        return min_total