class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        m = n // 2
        answer = []
        
        # Precompute prefix sums for each character
        prefix = [[0] * (n + 1) for _ in range(26)]
        for i in range(n):
            c = ord(s[i]) - ord('a')
            for j in range(26):
                prefix[j][i+1] = prefix[j][i]
            prefix[c][i+1] += 1
        
        # Precompute mismatch array and its prefix sum
        mismatch = [0] * n
        for i in range(m):
            j = n - 1 - i
            if s[i] != s[j]:
                mismatch[i] = 1
                mismatch[j] = 1
        prefix_mismatch = [0] * (n + 1)
        for i in range(n):
            prefix_mismatch[i+1] = prefix_mismatch[i] + mismatch[i]
        
        for q in queries:
            a, b, c, d = q
            valid = True
            
            # Check fixed pairs
            # i ranges not in A: [0, a-1] and [b+1, m-1]
            ranges_i = [(0, a-1), (b+1, m-1)]
            for (i_start, i_end) in ranges_i:
                if i_start > i_end:
                    continue
                j_low = n-1 - i_end
                j_high = n-1 - i_start
                # Check intersection with [m, c-1]
                start_j = max(j_low, m)
                end_j = min(j_high, c-1)
                if start_j <= end_j:
                    # Check if any mismatch in this j range
                    cnt = prefix_mismatch[end_j] - (prefix_mismatch[start_j-1] if start_j >0 else 0)
                    if cnt > 0:
                        valid = False
                        break
                # Check intersection with [d+1, n-1]
                start_j = max(j_low, d+1)
                end_j = min(j_high, n-1)
                if start_j <= end_j:
                    cnt = prefix_mismatch[end_j] - (prefix_mismatch[start_j-1] if start_j >0 else 0)
                    if cnt > 0:
                        valid = False
                        break
            if not valid:
                answer.append(False)
                continue
            
            # Compute required_A and required_B
            required_A = [0] * 26
            # A is [a, b], j ranges for i in A are [n-1 -b, n-1 -a]
            j_low_A = n-1 - b
            j_high_A = n-1 - a
            # Check intersection with [m, c-1]
            start_j = max(j_low_A, m)
            end_j = min(j_high_A, c-1)
            if start_j <= end_j:
                for c_id in range(26):
                    required_A[c_id] += prefix[c_id][end_j+1] - prefix[c_id][start_j]
            # Check intersection with [d+1, n-1]
            start_j = max(j_low_A, d+1)
            end_j = min(j_high_A, n-1)
            if start_j <= end_j:
                for c_id in range(26):
                    required_A[c_id] += prefix[c_id][end_j+1] - prefix[c_id][start_j]
            
            required_B = [0] * 26
            # B is [c, d], i ranges for j in B are [n-1 -d, n-1 -c]
            i_low_B = n-1 - d
            i_high_B = n-1 - c
            # Check intersection with [0, a-1]
            start_i = max(i_low_B, 0)
            end_i = min(i_high_B, a-1)
            if start_i <= end_i:
                for c_id in range(26):
                    required_B[c_id] += prefix[c_id][end_i+1] - prefix[c_id][start_i]
            # Check intersection with [b+1, m-1]
            start_i = max(i_low_B, b+1)
            end_i = min(i_high_B, m-1)
            if start_i <= end_i:
                for c_id in range(26):
                    required_B[c_id] += prefix[c_id][end_i+1] - prefix[c_id][start_i]
            
            # Get characters in A and B
            A_chars = [0] * 26
            for c_id in range(26):
                A_chars[c_id] = prefix[c_id][b+1] - prefix[c_id][a]
            B_chars = [0] * 26
            for c_id in range(26):
                B_chars[c_id] = prefix[c_id][d+1] - prefix[c_id][c]
            
            # Check if A has required_A
            for c_id in range(26):
                if A_chars[c_id] < required_A[c_id]:
                    valid = False
                    break
            if not valid:
                answer.append(False)
                continue
            
            # Check if B has required_B
            for c_id in range(26):
                if B_chars[c_id] < required_B[c_id]:
                    valid = False
                    break
            if not valid:
                answer.append(False)
                continue
            
            # Subtract required from A and B
            remaining_A = [max(0, A_chars[i] - required_A[i]) for i in range(26)]
            remaining_B = [max(0, B_chars[i] - required_B[i]) for i in range(26)]
            
            # Check even counts
            for i in range(26):
                total = remaining_A[i] + remaining_B[i]
                if total % 2 != 0:
                    valid = False
                    break
            answer.append(valid)
        
        return answer