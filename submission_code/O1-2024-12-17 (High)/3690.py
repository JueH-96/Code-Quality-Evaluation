class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        # Edge case: if string length <= 1, no operation can shorten the longest run
        if n <= 1:
            return n

        # Function to check if we can ensure that no run of identical bits
        # exceeds length L using at most numOps flips via a greedy approach.
        def canAchieve(L: int) -> bool:
            flips = 0
            run_char = None  # Tracks which character we are counting in the current run
            run_len = 0      # Length of the current run
            for ch in s:
                if ch == run_char:
                    run_len += 1
                    # If run_len exceeds L, we must flip the current bit
                    if run_len > L:
                        flips += 1
                        if flips > numOps:
                            return False
                        # Flip the current bit to the opposite character
                        run_char = '1' if run_char == '0' else '0'
                        run_len = 1
                else:
                    # We have a new run of a different character
                    run_char = ch
                    run_len = 1
            return True

        # We'll binary-search on the answer (the maximum allowed run length).
        left, right = 1, n
        answer = n
        while left <= right:
            mid = (left + right) // 2
            if canAchieve(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer