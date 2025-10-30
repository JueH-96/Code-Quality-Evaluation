import sys

def main():
    """
    Main function to solve the problem.
    """
    try:
        input = sys.stdin.readline
        N_str = input()
        if not N_str: return
        N = int(N_str)
        A_str = input()
        if not A_str: return
        A_list = list(map(int, A_str.split()))
    except (IOError, ValueError):
        return

    # Use 1-based indexing for arrays to align with problem statement logic
    A = [0] + A_list
    
    # --- Preprocessing ---
    # pos[v] stores a list of 1-based indices i where A[i] == v
    pos = [[] for _ in range(N + 2)] 
    for i in range(1, N + 1):
        pos[A[i]].append(i)
        
    # prev[i] stores the previous 1-based index of the value A[i]
    prev = [0] * (N + 1)
    last_pos = [0] * (N + 2)
    for i in range(1, N + 1):
        v = A[i]
        prev[i] = last_pos[v]
        last_pos[v] = i

    # --- S1 calculation ---
    # s1 = sum_{L,R} |U(L,R)|
    s1 = 0
    for i in range(1, N + 1):
        s1 += (i - prev[i]) * (N - i + 1)
        
    # --- S2 calculation ---
    # s2 = sum_{L,R} |{(u, u+1) | u, u+1 in U(L,R)}|
    s2 = 0
    total_subarrays = N * (N + 1) // 2

    def count_subarrays_in_gaps(p_list):
        if not p_list:
            return total_subarrays
        
        res = 0
        last = 0
        for p in p_list:
            d = p - last - 1
            if d > 0:
                res += d * (d + 1) // 2
            last = p
        
        d = N - last
        if d > 0:
            res += d * (d + 1) // 2
        return res

    def merge_sorted_lists(l1, l2):
        res = []
        i, j = 0, 0
        n1, n2 = len(l1), len(l2)
        while i < n1 and j < n2:
            if l1[i] < l2[j]:
                res.append(l1[i])
                i += 1
            elif l2[j] < l1[i]:
                res.append(l2[j])
                j += 1
            else:
                res.append(l1[i])
                i += 1
                j += 1
        
        res.extend(l1[i:])
        res.extend(l2[j:])
        return res

    # Iterate through all possible adjacent pairs of values (u, u+1)
    for u in range(1, N): # Pairs (u, u+1) require u <= N-1
        pos_u = pos[u]
        pos_u_plus_1 = pos[u + 1]
        
        if not pos_u or not pos_u_plus_1:
            continue
            
        count_not_u = count_subarrays_in_gaps(pos_u)
        count_not_u_plus_1 = count_subarrays_in_gaps(pos_u_plus_1)
        
        p_union = merge_sorted_lists(pos_u, pos_u_plus_1)
        count_not_both = count_subarrays_in_gaps(p_union)
        
        count_both = total_subarrays - count_not_u - count_not_u_plus_1 + count_not_both
        s2 += count_both

    # --- Final result ---
    print(s1 - s2)

if __name__ == "__main__":
    main()