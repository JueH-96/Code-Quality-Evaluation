class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        answer = 0
        
        # Enumerate all substrings by expanding the end index j from a start index i.
        for i in range(n):
            zero_count = 0
            one_count = 0
            for j in range(i, n):
                if s[j] == '0':
                    zero_count += 1
                else:
                    one_count += 1
                
                # If it satisfies at most k zeros or at most k ones, count it.
                if zero_count <= k or one_count <= k:
                    answer += 1
                else:
                    # If both exceed k, further extension won't satisfy the condition.
                    break
        
        return answer