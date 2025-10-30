import itertools
import math

class Solution:
    def generate_compositions(self, n, r):
        if r == 1:
            yield [n]
        else:
            for i in range(0, n+1):
                for comp in self.generate_compositions(n-i, r-1):
                    yield [i] + comp

    def countGoodIntegers(self, n: int, k: int) -> int:
        fact = [1] * (n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1] * i
        
        comps = list(self.generate_compositions(n, 10))
        ans = 0
        
        for comp in comps:
            if n % 2 == 0:
                valid_comp = True
                for count in comp:
                    if count % 2 != 0:
                        valid_comp = False
                        break
            else:
                count_odd = 0
                for count in comp:
                    if count % 2 == 1:
                        count_odd += 1
                valid_comp = (count_odd == 1)
            
            if not valid_comp:
                continue
            
            non_zero_exists = any(comp[d] > 0 for d in range(1, 10))
            if not non_zero_exists:
                continue
            
            if n % 2 == 0:
                sym_multiset = []
                for d in range(10):
                    count_here = comp[d] // 2
                    sym_multiset.extend([d] * count_here)
                
                perms_set = set(itertools.permutations(sym_multiset))
                found = False
                for p in perms_set:
                    if p[0] == 0:
                        continue
                    s = ''.join(str(x) for x in p) + ''.join(str(x) for x in p[::-1])
                    num = int(s)
                    if num % k == 0:
                        found = True
                        break
                
                if found:
                    total_perms = fact[n]
                    for count_val in comp:
                        total_perms //= fact[count_val]
                    
                    if comp[0] > 0:
                        new_comp = comp.copy()
                        new_comp[0] -= 1
                        subtract = fact[n-1]
                        for cnt in new_comp:
                            subtract //= fact[cnt]
                    else:
                        subtract = 0
                    ans += total_perms - subtract
            else:
                center_digit = None
                for d in range(10):
                    if comp[d] % 2 == 1:
                        center_digit = d
                        break
                sym_multiset = []
                for d in range(10):
                    count_here = comp[d] // 2
                    if d == center_digit:
                        count_here = (comp[d] - 1) // 2
                    sym_multiset.extend([d] * count_here)
                
                perms_set = set(itertools.permutations(sym_multiset))
                found = False
                for p in perms_set:
                    if len(p) > 0 and p[0] == 0:
                        continue
                    s = ''.join(str(x) for x in p) + str(center_digit) + ''.join(str(x) for x in p[::-1])
                    num = int(s)
                    if num % k == 0:
                        found = True
                        break
                
                if found:
                    total_perms = fact[n]
                    for count_val in comp:
                        total_perms //= fact[count_val]
                    
                    if comp[0] > 0:
                        new_comp = comp.copy()
                        new_comp[0] -= 1
                        subtract = fact[n-1]
                        for cnt in new_comp:
                            subtract //= fact[cnt]
                    else:
                        subtract = 0
                    ans += total_perms - subtract
        
        return ans