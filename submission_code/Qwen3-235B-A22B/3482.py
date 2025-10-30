class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        import sys
        import threading

        def run():
            mod = 10**9 + 7
            base = 911382629
            n = len(target)
            if n == 0:
                print(0)
                return

            # Determine the maximum length needed for pow_table
            max_L = 0
            for word in words:
                current_len = len(word)
                if current_len > max_L:
                    max_L = current_len
            pow_len = max(n, max_L)

            # Precompute powers of base modulo mod
            pow_table = [1] * (pow_len + 1)
            for i in range(1, pow_len + 1):
                pow_table[i] = (pow_table[i-1] * base) % mod

            # Compute prefix hashes for target
            prefix_hash = [0] * (n + 1)
            for i in range(n):
                prefix_hash[i+1] = (prefix_hash[i] * base + ord(target[i])) % mod

            # Build hash_map: for each L, map hash to minimal cost
            from collections import defaultdict
            hash_map = defaultdict(dict)  # hash_map[L][hash] = min cost

            for word, cost in zip(words, costs):
                L = len(word)
                # Compute hash for the word
                h = 0
                for ch in word:
                    h = (h * base + ord(ch)) % mod
                # Update hash_map for L and h
                if h in hash_map[L]:
                    if cost < hash_map[L][h]:
                        hash_map[L][h] = cost
                else:
                    hash_map[L][h] = cost

            # Collect all distinct lengths present in hash_map
            distinct_lengths = list(hash_map.keys())

            # Initialize DP array
            INF = float('inf')
            dp = [INF] * (n + 1)
            dp[0] = 0

            # Process each position
            for i in range(n + 1):
                if dp[i] == INF:
                    continue
                for L in distinct_lengths:
                    if i + L > n:
                        continue
                    # Compute hash of target[i:i+L]
                    current_hash = (prefix_hash[i + L] - (prefix_hash[i] * pow_table[L]) % mod) % mod
                    current_hash = (current_hash + mod) % mod  # Ensure non-negative
                    # Check if current_hash exists in hash_map[L]
                    if current_hash in hash_map[L]:
                        min_cost = hash_map[L][current_hash]
                        if dp[i] + min_cost < dp[i + L]:
                            dp[i + L] = dp[i] + min_cost

            result = dp[n] if dp[n] != INF else -1
            print(result)

        threading.Thread(target=run).start()