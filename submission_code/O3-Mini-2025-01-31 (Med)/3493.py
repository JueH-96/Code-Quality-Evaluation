class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        # record positions of all '1's in the string
        pos = [i for i,ch in enumerate(s) if ch == '1']
        m = len(pos)
        # if there are no '1's, or the string already has
        # all zeros then no operation is possible.
        if m == 0:
            return 0
        # if there is only one '1', then at most one operation
        # (it “jumps” to the end) is possible if it is not already at n-1.
        if m == 1:
            return 1 if pos[0] < n - 1 else 0

        # We simulate “moving” only the positions of the 1’s.
        # When a non–last 1 (index i, with i in 0..m-2) is eligible to move if:
        #    pos[i] < pos[i+1] - 1.
        # When such a move is performed the allowed jump is forced:
        #    pos[i] becomes pos[i+1] - 1.
        #
        # We use a deque to hold the indices (from 0 to m-2) that are eligible.
        from collections import deque
        dq = deque()
        in_dq = [False] * m  # to mark which indices are in our deque
        
        # initialize: for each non–last index, add it if eligible.
        for i in range(m - 1):
            if pos[i] < pos[i+1] - 1:
                dq.append(i)
                in_dq[i] = True
        
        # Also keep a flag for the very last “1”
        # (the legal move on the last 1 is allowed if pos[m-1] < n-1).
        last_ready = (pos[m-1] < n - 1)
        
        ops = 0
        # Simulation loop: while some move is possible.
        while True:
            if dq:
                i = dq.popleft()
                in_dq[i] = False
                # check eligibility (it might have changed)
                if pos[i] < pos[i+1] - 1:
                    # perform the forced move: the 1 at index i jumps to (pos[i+1] - 1)
                    pos[i] = pos[i+1] - 1
                    ops += 1
                    # Now, because pos[i] increased, its left–neighbor (if any)
                    # might now have room to move one step (if pos[i-1] < pos[i] - 1)
                    if i - 1 >= 0:
                        if pos[i-1] < pos[i] - 1 and not in_dq[i-1]:
                            dq.append(i-1)
                            in_dq[i-1] = True
                continue
            elif last_ready and pos[m-1] < n - 1:
                # if no non–last 1 is eligible and the last one is not at the end,
                # then perform its (forced) jump.
                pos[m-1] = n - 1
                ops += 1
                last_ready = False  # once at end, last one can no longer move.
                # Now the gap between the last two 1’s may open up; check index m-2
                i = m - 2
                if pos[i] < pos[m-1] - 1 and not in_dq[i]:
                    dq.append(i)
                    in_dq[i] = True
                continue
            else:
                break
        return ops

# For testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.maxOperations("1001101"))   # Expected output: 4
    # Example 2:
    print(sol.maxOperations("00111"))     # Expected output: 0
    # Some more tests:
    print(sol.maxOperations("1010"))      # one possible maximum is 3
    print(sol.maxOperations("1100"))      # expected: 2
    print(sol.maxOperations("10101"))     # expected: 3
    print(sol.maxOperations("1000"))      # expected: 1