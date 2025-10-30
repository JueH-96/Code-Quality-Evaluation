class Solution:
    def minOperations(self, n: int, m: int) -> int:
        def cost(s):
            return sum([int(c) for c in str(s)])
        
        def nextLarger(x):
            s, ns, ops = [], [], 0
            for ch in str(x)[:-1]:
                s.append(int(ch))
            c = int(str(x)[-1]) + 1
            if c == 10: c = 0
            s.append(c)
            for i in reversed(range(len(s))):
                d = s[i]
                if (i == 0 and d != 9) or (i > 0 and (s[i - 1] != 9 or d < 8)):
                    ns.insert(0, d)
                elif d != 9:
                    ns.insert(0, d + 1)
                    ops += 1
                elif i > 0 and s[i - 1] == 9:
                    prev = int(str(x)[:i] + '0' + str(x)[i + 1:])
                    nxt = nextSmaller(prev)
                    if nxt:
                        return nxt, cost(prev) + cost(nxt) - 2 * cost(x) + 1
                    else:
                        return None, None
                elif i == 0:
                    return None, None
            
            return int(''.join([str(d) for d in ns])), cost(int(''.join([str(d) for d in ns]))) - cost(x) + ops
        
        def nextSmaller(x):
            s, ns, ops = [], [], 0
            if x == 0:
                return None
            for i in reversed(range(len(str(x)))):
                d = int(str(x)[i])
                if i == 0 or d != 1:
                    ns.insert(0, d)
                elif i > 0 and (s[-1] != 1 or int(str(x)[i - 1]) > 1):
                    ns.insert(0, d - 1)
                    ops += 1
                elif i > 0 and s[-1] == 1 and int(str(x)[i - 1]) == 1:
                    prev = int(str(x)[:i] + '9' + str(x)[i + 1:])
                    nxt = nextLarger(prev)
                    if nxt:
                        return nxt, cost(prev) + cost(nxt) - 2 * cost(x) + 1
                    else:
                        return None, None
            
            return int(''.join([str(d) for d in ns])), cost(int(''.join([str(d) for d in ns]))) - cost(x) + ops
        
        def smallestNonPrimeGeq(x):
            if x == 1:
                return 4, 4
            if is_prime(x):
                res, cost = nextLarger(x)
                if res:
                    return res, cost
                return None, None
            return x, cost(x)
        
        def is_prime(n):
            if n == 2 or n == 3: return True
            if n % 2 == 0 or n < 2: return False
            for i in range(3, int(n ** 0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        n_cost = cost(n)
        m_cost = cost(m)
        n, n_cost, _ = smallestNonPrimeGeq(n) or (None, None, None)
        m, m_cost, _ = smallestNonPrimeGeq(m) or (None, None, None)

        if n == None or m == None or n_cost == None or m_cost == None: 
            return -1

        if n_cost == m_cost and n == m:
            return n_cost
        
        cur, cur_cost = n, n_cost
        ops = 0
        while cur != m:
            if cur_cost > m_cost:
                nxt = nextSmaller(cur)
            else:
                nxt = nextLarger(cur)
            
            if nxt[0] == None:
                return -1
            cur_cost += nxt[1]
            cur = nxt[0]
            ops += 1
        
        # consider cascade of 9s to 0s and 0s to 9s as 1 move (yes/no indistinguishable later)
        return n_cost + (m_cost - n_cost) * ops + m_cost