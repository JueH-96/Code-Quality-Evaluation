class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        
        # First, create an array of indices where s[i]=='0'
        pos = [i for i, ch in enumerate(s) if ch == '0']
        total = 0
        
        # 1. Count substrings with 0 zeros.
        # They are contiguous sequences of 1's. We can count these by scanning.
        cnt = 0
        for ch in s:
            if ch == '1':
                cnt += 1
            else:
                # when a zero is encountered, count the substrings before the zero
                total += cnt * (cnt + 1) // 2
                cnt = 0
        total += cnt * (cnt + 1) // 2  # update for final segment
        
        # 2. Count substrings with at least 1 zero.
        # Let k be number of zeros in the substring.
        m = len(pos)
        # For each block of k consecutive zeros in 'pos'
        # Only consider values of k such that k^2 + k <= n.
        # (For larger k, substring would have to be really long and there might be no candidates.)
        max_k = m  # k cannot exceed the total count of zeros in s.
        # But we limit further by k^2+k <= n.
        while max_k > 0 and max_k * (max_k + 1) > n:
            max_k -= 1

        # For each possible k from 1 to max_k, consider blocks of k zeros.
        for k in range(1, max_k + 1):
            T = k * (k + 1)  # required minimum length L >= k^2+k
            # Enumerate all blocks (i.e. contiguous groups in the pos array) with exactly k zeros.
            for i in range(0, m - k + 1):
                # the block of zeros is from pos[i] to pos[i+k-1]
                # Determine the left boundary A and right boundary B.
                A = pos[i - 1] + 1 if i > 0 else 0
                B = pos[i + k] - 1 if (i + k) < m else n - 1
                
                # For each possible starting index s in [A, pos[i]]
                # The minimal allowed endpoint is max(pos[i+k-1], s + T - 1).
                # So for s in [A, pos[i]], add number of endpoints e in [max(pos[i+k-1], s + T - 1), B].
                left_choice_max = pos[i]
                for s_idx in range(A, left_choice_max + 1):
                    # minimal e
                    e_min = s_idx + T - 1
                    if e_min < pos[i + k - 1]:
                        e_min = pos[i + k - 1]
                    if e_min > B:
                        continue
                    total += (B - e_min + 1)
        return total