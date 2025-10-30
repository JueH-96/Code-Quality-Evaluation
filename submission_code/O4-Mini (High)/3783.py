from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # We only care about counts up to k, so cap factorials at k+1
        cap = k + 1
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i
            if fact[i] > cap:
                fact[i] = cap

        # Initialize the list of available numbers and odd/even counts
        avail = list(range(1, n + 1))
        o_rem = (n + 1) // 2
        e_rem = n // 2

        res = []
        parity_req = None  # no parity requirement for the first position

        for pos in range(n):
            found = False
            # Try each candidate in lex order
            for x in avail:
                p = x & 1  # 1 if odd, 0 if even
                # Enforce alternating parity after the first position
                if pos > 0 and p != parity_req:
                    continue

                # Compute remaining odd/even if we pick x
                o_next = o_rem - p
                e_next = e_rem - (1 - p)
                L = n - pos - 1  # length of the suffix

                # Count how many valid suffixes follow
                if L == 0:
                    cnt = 1
                else:
                    next_parity = 1 - p
                    # Number of odd slots needed in the suffix
                    odd_req = (L + 1) // 2 if next_parity == 1 else L // 2
                    if o_next != odd_req:
                        cnt = 0
                    else:
                        cnt = fact[o_next] * fact[e_next]
                        if cnt > k:
                            cnt = cap

                # If the k-th permutation is not in this block, skip it
                if cnt < k:
                    k -= cnt
                    continue

                # Otherwise choose x and move on
                res.append(x)
                avail.remove(x)
                o_rem, e_rem = o_next, e_next
                parity_req = 1 - p
                found = True
                break

            if not found:
                # k is too large; no such permutation
                return []

        return res