class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        L = len(s)
        max_freq = max(freq.values()) if freq else 0
        min_cost = float('inf')
        
        # Iterate K from 1 to max_freq + 2 to cover possible cases
        for K in range(1, max_freq + 2):
            # Consider M values around the possible range
            M_min = max(1, (L - 2) // K) if K > 0 else 1
            M_max = (L + 2) // K if K > 0 else L
            if K == 0:
                continue  # K cannot be zero
            for M in range(M_min, M_max + 1):
                target_length = K * M
                insert_delete_cost = abs(L - target_length)
                costs = []
                for d in 'abcdefghijklmnopqrstuvwxyz':
                    # Calculate cost to get K occurrences of d
                    cost = 0
                    # Use original d's
                    take_d = min(freq.get(d, 0), K)
                    remaining = K - take_d
                    # Use d-1's if possible
                    if remaining > 0 and d > 'a':
                        prev = chr(ord(d) - 1)
                        take_prev = min(freq.get(prev, 0), remaining)
                        cost += take_prev * 1
                        remaining -= take_prev
                    # Use insertions for the remaining
                    cost += remaining * 2
                    costs.append(cost)
                # Select M characters with the smallest costs
                costs.sort()
                if len(costs) < M:
                    continue  # Not enough characters to form M targets
                total_modify_cost = sum(costs[:M])
                total_cost = total_modify_cost + insert_delete_cost
                if total_cost < min_cost:
                    min_cost = total_cost
        return min_cost