import sys
import heapq
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    L_count = int(data[2])
    a = list(map(int, data[3:3+n]))
    b = list(map(int, data[3+n:3+n+m]))
    forbidden_pairs = []
    idx = 3 + n + m
    for i in range(L_count):
        c = int(data[idx]); d = int(data[idx+1])
        idx += 2
        forbidden_pairs.append((c, d))
    
    max_a = max(a) if a else 0
    max_b = max(b) if b else 0

    max_a_indices = [i for i, x in enumerate(a) if x == max_a]
    max_b_indices = [j for j, x in enumerate(b) if x == max_b]
    max_a_set = set(max_a_indices)
    max_b_set = set(max_b_indices)
    
    total_pairs = len(max_a_indices) * len(max_b_indices)
    count = 0
    for (c, d) in forbidden_pairs:
        c0 = c - 1
        d0 = d - 1
        if c0 in max_a_set and d0 in max_b_set:
            count += 1
            if count == total_pairs:
                break
                
    if count < total_pairs:
        print(max_a + max_b)
        return
        
    candidate_main = set()
    candidate_side = set()
    forbidden_side_for_main = defaultdict(set)
    
    for (c, d) in forbidden_pairs:
        c0 = c - 1
        d0 = d - 1
        candidate_main.add(c0)
        candidate_side.add(d0)
        forbidden_side_for_main[c0].add(d0)
        
    candidate_ans = 0
    
    non_main_vals = [a[i] for i in range(n) if i not in candidate_main]
    if non_main_vals:
        max_non_main = max(non_main_vals)
        candidate_ans = max(candidate_ans, max_non_main + max_b)
        
    non_side_vals = [b[j] for j in range(m) if j not in candidate_side]
    if non_side_vals:
        max_non_side = max(non_side_vals)
        candidate_ans = max(candidate_ans, max_a + max_non_side)
        
    if candidate_main and candidate_side:
        side_list = []
        for j in candidate_side:
            side_list.append((b[j], j))
        side_list.sort(key=lambda x: x[0], reverse=True)
        
        heap = []
        for i in candidate_main:
            total_val = a[i] + side_list[0][0]
            heapq.heappush(heap, (-total_val, i, 0))
            
        found_candidate3 = False
        candidate3_val = 0
        while heap:
            neg_val, i, ptr = heapq.heappop(heap)
            total_val = -neg_val
            j_index = side_list[ptr][1]
            
            if i in forbidden_side_for_main and j_index in forbidden_side_for_main[i]:
                if ptr + 1 < len(side_list):
                    next_val = a[i] + side_list[ptr+1][0]
                    heapq.heappush(heap, (-next_val, i, ptr+1))
            else:
                candidate3_val = total_val
                found_candidate3 = True
                break
                
        if found_candidate3:
            candidate_ans = max(candidate_ans, candidate3_val)
            
    print(candidate_ans)

if __name__ == '__main__':
    main()