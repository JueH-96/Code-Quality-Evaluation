import sys, math

MOD = 998244353

def prefix_next(v, mask, shifts):
    for s in shifts:
        v ^= ( (v << s) & mask)
    return v & mask

class Fenwick:
    def __init__(self, n):
        self.n = n
        self.bit = [0]*(n+1)
    def add(self, i, delta=1):
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i
    def sum(self, i):
        i +=1
        res=0
        while i:
            res += self.bit[i]
            i -= i & -i
        return res

def main():
    import sys
    data = list(map(int, sys.stdin.buffer.read().split()))
    N, M = data[0], data[1]
    ptr = 2
    values = [0]*N
    value_to_indices = dict()
    for i in range(N):
        val = 0
        for b in range(M):
            if data[ptr]:
                val |= 1 << b
            ptr += 1
        values[i] = val
        value_to_indices.setdefault(val, []).append(i)
    mask = (1<<M) - 1 if M!=0 else 0
    shifts = []
    s = 1
    while s < M:
        shifts.append(s)
        s <<=1
    pos_of_idx = [0]*N
    cycle_of_idx = [0]*N
    cycles_indices = []
    cycles_len = []
    visited_inputs = set()
    cycle_id = 0
    for val in value_to_indices.keys():
        if val in visited_inputs:
            continue
        indices_list = []
        pos = 0
        current = val
        while True:
            if current in value_to_indices and current not in visited_inputs:
                for idx in value_to_indices[current]:
                    pos_of_idx[idx] = pos
                    cycle_of_idx[idx] = cycle_id
                    indices_list.append(idx)
                visited_inputs.add(current)
            nxt = current
            for s in shifts:
                nxt ^= ((nxt << s) & mask)
            nxt &= mask
            pos += 1
            if nxt == val:
                break
            current = nxt
        cycles_indices.append(indices_list)
        cycles_len.append(pos)
        cycle_id += 1
    total = 0
    for cid, idx_list in enumerate(cycles_indices):
        k = len(idx_list)
        if k <= 1:
            continue
        idx_list.sort()
        len_cycle = cycles_len[cid]
        pref_sum = 0
        sum_diff = 0
        pos_vals = [pos_of_idx[idx] for idx in idx_list]
        # compression for inversion
        uniq = sorted(set(pos_vals))
        rank = {v:i for i,v in enumerate(uniq)}
        bit = Fenwick(len(uniq))
        inversion = 0
        seen = 0
        for j, p in enumerate(pos_vals):
            sum_diff += p*j - pref_sum
            pref_sum += p
            r = rank[p]
            lesser_equal = bit.sum(r)
            inversion += seen - lesser_equal
            bit.add(r,1)
            seen +=1
        contribution = (sum_diff + inversion * len_cycle) % MOD
        total = (total + contribution) % MOD
    print(total % MOD)

if __name__ == "__main__":
    main()