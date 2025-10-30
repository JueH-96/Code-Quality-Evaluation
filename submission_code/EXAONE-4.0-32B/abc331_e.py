import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    L = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    b = [int(next(it)) for _ in range(m)]
    forbidden_pairs = []
    for _ in range(L):
        c = int(next(it))
        d = int(next(it))
        forbidden_pairs.append((c-1, d-1))
    
    max_a = max(a)
    max_b = max(b)
    indices_A = [i for i in range(n) if a[i] == max_a]
    indices_B = [j for j in range(m) if b[j] == max_b]
    set_A = set(indices_A)
    set_B = set(indices_B)
    
    count_forbidden_in_S = 0
    for (i, j) in forbidden_pairs:
        if i in set_A and j in set_B:
            count_forbidden_in_S += 1
            
    total_pairs_in_S = len(indices_A) * len(indices_B)
    if count_forbidden_in_S < total_pairs_in_S:
        print(max_a + max_b)
        return
        
    sorted_a = sorted(a, reverse=True)
    sorted_b = sorted(b, reverse=True)
    
    if n <= L:
        threshold_main = sorted_a[-1]
    else:
        threshold_main = sorted_a[L]
        
    if m <= L:
        threshold_side = sorted_b[-1]
    else:
        threshold_side = sorted_b[L]
        
    X = [i for i in range(n) if a[i] >= threshold_main]
    Y = [j for j in range(m) if b[j] >= threshold_side]
    
    forbidden_side_by_main = [set() for _ in range(n)]
    forbidden_main_by_side = [set() for _ in range(m)]
    for (i, j) in forbidden_pairs:
        forbidden_side_by_main[i].add(j)
        forbidden_main_by_side[j].add(i)
        
    sorted_side_list = [(b[j], j) for j in range(m)]
    sorted_side_list.sort(key=lambda x: x[0], reverse=True)
    
    sorted_main_list = [(a[i], i) for i in range(n)]
    sorted_main_list.sort(key=lambda x: x[0], reverse=True)
    
    candidate_ans = 0
    for i in X:
        for k in range(m):
            val, j_idx = sorted_side_list[k]
            if j_idx not in forbidden_side_by_main[i]:
                total = a[i] + val
                if total > candidate_ans:
                    candidate_ans = total
                break
                
    for j in Y:
        for k in range(n):
            val, i_idx = sorted_main_list[k]
            if i_idx not in forbidden_main_by_side[j]:
                total = val + b[j]
                if total > candidate_ans:
                    candidate_ans = total
                break
                    
    print(candidate_ans)

if __name__ == "__main__":
    main()