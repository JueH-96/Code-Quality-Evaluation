import sys
import functools
sys.setrecursionlimit(10000)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    P = []
    index = 1
    for i in range(n):
        x = int(data[index]); y = int(data[index+1]); index += 2
        P.append((x, y, i, 'P'))
    Q = []
    for i in range(n):
        x = int(data[index]); y = int(data[index+1]); index += 2
        Q.append((x, y, i, 'Q'))
    
    matches = solve(P, Q)
    if matches is None:
        print(-1)
    else:
        res = [0] * n
        for (p_idx, q_idx) in matches:
            res[p_idx] = q_idx + 1
        print(" ".join(map(str, res)))

def solve(P_list, Q_list):
    if len(P_list) == 0 and len(Q_list) == 0:
        return []
    all_points = P_list + Q_list
    O = min(all_points, key=lambda p: (p[0], p[1]))
    
    if O[3] == 'P':
        new_P_list = [p for p in P_list if p != O]
        new_Q_list = [p for p in Q_list]
        remaining = new_P_list + new_Q_list
        
        if not remaining:
            return None
            
        def cmp_func(a, b):
            ax = a[0] - O[0]; ay = a[1] - O[1]
            bx = b[0] - O[0]; by = b[1] - O[1]
            cross = ax * by - ay * bx
            if cross > 0:
                return -1
            elif cross < 0:
                return 1
            else:
                da = ax*ax + ay*ay
                db = bx*bx + by*by
                if da < db:
                    return -1
                else:
                    return 1
                    
        try:
            remaining_sorted = sorted(remaining, key=functools.cmp_to_key(cmp_func))
        except Exception as e:
            return None
        
        balance = 0
        candidate = None
        candidate_index = -1
        for i, p in enumerate(remaining_sorted):
            if p[3] == 'Q' and balance == 0:
                candidate = p
                candidate_index = i
                break
                
            if p[3] == 'P':
                balance += 1
            else:
                balance -= 1
                
        if candidate is None:
            return None
            
        set1 = remaining_sorted[:candidate_index]
        set2 = remaining_sorted[candidate_index+1:]
        
        set1_P = [p for p in set1 if p[3]=='P']
        set1_Q = [p for p in set1 if p[3]=='Q']
        set2_P = [p for p in set2 if p[3]=='P']
        set2_Q = [p for p in set2 if p[3]=='Q']
        
        match1 = solve(set1_P, set1_Q)
        match2 = solve(set2_P, set2_Q)
        if match1 is None or match2 is None:
            return None
            
        return [(O[2], candidate[2])] + match1 + match2
        
    else:
        new_Q_list = [p for p in Q_list if p != O]
        new_P_list = [p for p in P_list]
        remaining = new_P_list + new_Q_list
        
        if not remaining:
            return None
            
        def cmp_func(a, b):
            ax = a[0] - O[0]; ay = a[1] - O[1]
            bx = b[0] - O[0]; by = b[1] - O[1]
            cross = ax * by - ay * bx
            if cross > 0:
                return -1
            elif cross < 0:
                return 1
            else:
                da = ax*ax + ay*ay
                db = bx*bx + by*by
                if da < db:
                    return -1
                else:
                    return 1
                    
        try:
            remaining_sorted = sorted(remaining, key=functools.cmp_to_key(cmp_func))
        except:
            return None
            
        balance = 0
        candidate = None
        candidate_index = -1
        for i, p in enumerate(remaining_sorted):
            if p[3] == 'P' and balance == 0:
                candidate = p
                candidate_index = i
                break
                
            if p[3] == 'P':
                balance += 1
            else:
                balance -= 1
                
        if candidate is None:
            return None
            
        set1 = remaining_sorted[:candidate_index]
        set2 = remaining_sorted[candidate_index+1:]
        
        set1_P = [p for p in set1 if p[3]=='P']
        set1_Q = [p for p in set1 if p[3]=='Q']
        set2_P = [p for p in set2 if p[3]=='P']
        set2_Q = [p for p in set2 if p[3]=='Q']
        
        match1 = solve(set1_P, set1_Q)
        match2 = solve(set2_P, set2_Q)
        if match1 is None or match2 is None:
            return None
            
        return [(candidate[2], O[2])] + match1 + match2

if __name__ == '__main__':
    main()