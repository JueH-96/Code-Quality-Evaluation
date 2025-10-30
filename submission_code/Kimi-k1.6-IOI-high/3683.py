class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        K = numFriends
        if K == 1:
            return word
        max_str = ""
        
        # Case 1: Starts at the beginning (i=0)
        max_j_case1 = n - K
        if max_j_case1 >= 0:
            for j in range(0, max_j_case1 + 1):
                current = word[0:j+1]
                if current > max_str:
                    max_str = current
        
        # Case 2: Ends at the last character (j = n-1)
        min_i_case2 = K - 1
        if min_i_case2 < n:
            for i in range(min_i_case2, n):
                current = word[i:]
                if current > max_str:
                    max_str = current
        
        # Case 3: Middle parts when K >=3
        if K >= 3:
            for i in range(1, n-1):
                for j in range(i, n-1):
                    L = (n - 1) - j  # Length of the part after j
                    K_minus_1 = K - 1
                    a_min = max(1, K_minus_1 - L)
                    a_max = min(i, K_minus_1 - 1)
                    if a_min <= a_max:
                        current = word[i:j+1]
                        if current > max_str:
                            max_str = current
        
        return max_str