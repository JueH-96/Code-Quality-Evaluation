class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        mismatches = []
        for i in range(n):
            if s1[i] != s2[i]:
                mismatches.append(i)
        M = len(mismatches)
        if M % 2 != 0:
            return -1
        total_cost = 0
        i = 0
        isolated_mismatches = []
        while i < M:
            run_start = i
            # Find the length of the run of mismatches
            while i + 1 < M and mismatches[i + 1] == mismatches[i] +1:
                i +=1
            run_end = i
            L = run_end - run_start +1
            if L ==1:
                isolated_mismatches.append(mismatches[i])
            elif L % 2 == 0:
                total_cost += (L //2) *1
            else:  # L >=3 and odd
                total_cost += ((L -2) // 2)*1 + x
            i +=1
        # Pair up isolated mismatches
        K = len(isolated_mismatches)
        if K %2 != 0:
            return -1
        total_cost += (K //2) * x
        return total_cost