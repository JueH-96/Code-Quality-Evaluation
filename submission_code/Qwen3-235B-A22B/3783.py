from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        from functools import lru_cache
        import sys
        sys.setrecursionlimit(1000000)
        
        evens = [i for i in range(1, n+1) if i % 2 == 0]
        odds = [i for i in range(1, n+1) if i % 2 == 1]
        even_count = len(evens)
        odd_count = len(odds)
        
        @lru_cache(maxsize=None)
        def count(E, O, req_parity):
            if E < 0 or O < 0:
                return 0
            if E == 0 and O == 0:
                return 1
            if req_parity == 0:
                if E == 0:
                    return 0
                return E * count(E-1, O, 1)
            else:
                if O == 0:
                    return 0
                return O * count(E, O-1, 0)
        
        permutation = []
        used = [False] * (n + 2)  # 1-based indexing
        
        for step in range(n):
            if step == 0:
                current_prev_parity = None
                found = False
                for x in range(1, n+1):
                    if used[x]:
                        continue
                    x_parity = 0 if (x % 2 == 0) else 1
                    rem_even = even_count - (1 if x_parity == 0 else 0)
                    rem_odd = odd_count - (1 if x_parity == 1 else 0)
                    next_req = 1 - x_parity
                    c = count(rem_even, rem_odd, next_req)
                    if k > c:
                        k -= c
                        continue
                    else:
                        permutation.append(x)
                        used[x] = True
                        current_even_count = rem_even
                        current_odd_count = rem_odd
                        current_prev_parity = x_parity
                        found = True
                        break
                if not found:
                    return []
            else:
                required_parity = 1 - current_prev_parity
                found = False
                for x in range(1, n+1):
                    if used[x]:
                        continue
                    x_parity = 0 if (x % 2 == 0) else 1
                    if x_parity != required_parity:
                        continue
                    rem_even = current_even_count - ( (x_parity == 0) )
                    rem_odd = current_odd_count - ( (x_parity == 1) )
                    next_req = 1 - required_parity
                    c = count(rem_even, rem_odd, next_req)
                    if k > c:
                        k -= c
                        continue
                    else:
                        permutation.append(x)
                        used[x] = True
                        current_even_count = rem_even
                        current_odd_count = rem_odd
                        current_prev_parity = x_parity
                        found = True
                        break
                if not found:
                    return []
        return permutation