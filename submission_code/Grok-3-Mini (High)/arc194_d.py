import sys
import collections
sys.setrecursionlimit(100000)

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = data[index]

# Compute matching parentheses
match_arr = [-1] * N
stack = []
for i in range(N):
    if S[i] == '(':
        stack.append(i)
    else:  # ')'
        open_idx = stack.pop()
        match_arr[open_idx] = i
        match_arr[i] = open_idx  # Optional, but useful for symmetry

MOD = 998244353
MAX_FACT = 5005
fact = [1] * MAX_FACT
for i in range(1, MAX_FACT):
    fact[i] = (fact[i-1] * i) % MOD
inv_fact = [0] * MAX_FACT
for i in range(MAX_FACT):
    inv_fact[i] = pow(fact[i], MOD - 2, MOD)

def compute_subtree(L, R):
    M = match_arr[L]
    if M == R:
        child_start = L + 1
        child_end = R - 1
    else:
        child_start = L
        child_end = R
    if child_start > child_end:
        children_intervals = []
    else:
        children_intervals = []
        pos = child_start
        end_pos = child_end
        while pos <= end_pos:
            M_pos = match_arr[pos]
            children_intervals.append((pos, M_pos))
            pos = M_pos + 1
    m = len(children_intervals)
    if m == 0:
        multinomial = 1
        prod_children = 1
        total_prod = (multinomial * prod_children) % MOD
        node_string = S[L:R+1]
        return total_prod, node_string
    else:
        child_results = [compute_subtree(cl, cr) for cl, cr in children_intervals]
        child_prods = [res[0] for res in child_results]
        child_strings = [res[1] for res in child_results]
        counter_obj = collections.Counter(child_strings)
        num = fact[m]
        den_inv_prod = 1
        for freq in counter_obj.values():
            den_inv_prod = (den_inv_prod * inv_fact[freq]) % MOD
        multinomial = (num * den_inv_prod) % MOD
        prod_children = 1
        for cp in child_prods:
            prod_children = (prod_children * cp) % MOD
        total_prod = (multinomial * prod_children) % MOD
        node_string = S[L:R+1]
        return total_prod, node_string

answer, _ = compute_subtree(0, N-1)
print(answer)