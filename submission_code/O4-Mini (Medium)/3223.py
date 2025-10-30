class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        if n == 0 or k > n:
            return 0
        
        # Precompute prefix sums of "bad" adjacent pairs:
        # bad[i] = 1 if |word[i] - word[i+1]| > 2, else 0
        # prefix_bad[j] = sum of bad[0..j-1]
        prefix_bad = [0] * (n + 1)
        for i in range(n):
            add = 0
            if i < n - 1 and abs(ord(word[i]) - ord(word[i+1])) > 2:
                add = 1
            prefix_bad[i+1] = prefix_bad[i] + add
        
        res = 0
        # We only need to consider substrings whose length L = m * k,
        # where m = 1..min(26, n//k)
        max_m = min(26, n // k)
        # We'll reuse a counts array for each m
        for m in range(1, max_m + 1):
            L = m * k
            if L > n:
                break
            
            counts = [0] * 26
            num_nonzero = 0    # how many chars have count > 0
            num_eq_k = 0       # how many chars have count == k
            
            # helper to add a character
            def add_char(c):
                nonlocal num_nonzero, num_eq_k
                idx = ord(c) - ord('a')
                old = counts[idx]
                # if it was exactly k, we'll lose that upon increment
                if old == k:
                    num_eq_k -= 1
                new = old + 1
                # if it was zero, now becomes nonzero
                if old == 0:
                    num_nonzero += 1
                # if it reaches k, count that
                if new == k:
                    num_eq_k += 1
                counts[idx] = new
            
            # helper to remove a character
            def remove_char(c):
                nonlocal num_nonzero, num_eq_k
                idx = ord(c) - ord('a')
                old = counts[idx]
                # if it was exactly k, we'll lose that upon decrement
                if old == k:
                    num_eq_k -= 1
                new = old - 1
                # if it drops to zero, one fewer nonzero
                if new == 0:
                    num_nonzero -= 1
                # if it becomes exactly k, count that
                if new == k:
                    num_eq_k += 1
                counts[idx] = new
            
            # build the first window [0..L-1]
            for i in range(L):
                add_char(word[i])
            
            # slide over all windows of length L
            for start in range(0, n - L + 1):
                end = start + L - 1
                # check counts condition
                if num_nonzero == m and num_eq_k == m:
                    # check adjacency condition via prefix_bad
                    if prefix_bad[end] - prefix_bad[start] == 0:
                        res += 1
                # slide: remove start, add start+L
                if start + L < n:
                    remove_char(word[start])
                    add_char(word[start + L])
        
        return res