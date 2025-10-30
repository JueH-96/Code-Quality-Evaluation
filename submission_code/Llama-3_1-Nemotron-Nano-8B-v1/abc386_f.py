def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    S = input[1]
    T = input[2]
    
    n = len(S)
    m = len(T)
    
    if abs(n - m) > K:
        print("No")
        return
    
    def is_possible(s, t, max_edits):
        len_s = len(s)
        len_t = len(t)
        if len_s > len_t:
            return is_possible(t, s, max_edits)
        
        prev_row = [float('inf')] * (2 * max_edits + 1)
        prev_row[0] = 0
        
        for i in range(len_s):
            curr_row = [float('inf')] * (2 * max_edits + 1)
            for d in range(-max_edits, max_edits + 1):
                if prev_row[d + max_edits] == float('inf'):
                    continue
                j = i - d
                if j < 0 or j > len_t:
                    continue
                
                if j < len_t and i < len_s:
                    cost_sub = prev_row[d + max_edits]
                    if s[i] != t[j]:
                        cost_sub += 1
                    new_d = d
                    if abs(new_d) <= max_edits:
                        if cost_sub < curr_row[new_d + max_edits]:
                            curr_row[new_d + max_edits] = cost_sub
                
                if i < len_s:
                    new_d_del = d + 1
                    if new_d_del <= max_edits:
                        cost_del = prev_row[d + max_edits] + 1
                        if cost_del < curr_row[new_d_del + max_edits]:
                            curr_row[new_d_del + max_edits] = cost_del
                
                if j < len_t:
                    new_d_ins = d - 1
                    if new_d_ins >= -max_edits:
                        cost_ins = prev_row[d + max_edits] + 1
                        if cost_ins < curr_row[new_d_ins + max_edits]:
                            curr_row[new_d_ins + max_edits] = cost_ins
            
            prev_row = curr_row
        
        for d in range(-max_edits, max_edits + 1):
            if prev_row[d + max_edits] != float('inf'):
                j = len_s - d
                if j > len_t:
                    continue
                remaining = len_t - j
                if prev_row[d + max_edits] + remaining <= max_edits:
                    return True
        return False
    
    print("Yes" if is_possible(S, T, K) else "No")

if __name__ == "__main__":
    main()