import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+2*n]))
    
    S_candidate = None
    for i in range(n):
        if A[i] != -1 and B[i] != -1:
            s0 = A[i] + B[i]
            if S_candidate is None:
                S_candidate = s0
            else:
                if S_candidate != s0:
                    print("No")
                    return
                    
    X_known = [a for a in A if a != -1]
    Y_known = [b for b in B if b != -1]
    c_A = n - len(X_known)
    c_B = n - len(Y_known)
    
    if S_candidate is not None:
        S = S_candidate
        for i in range(n):
            if A[i] != -1 and B[i] == -1:
                if S < A[i]:
                    print("No")
                    return
            if A[i] == -1 and B[i] != -1:
                if S < B[i]:
                    print("No")
                    return
        
        T_list = [S - a for a in X_known]
        freq_T = defaultdict(int)
        for x in T_list:
            freq_T[x] += 1
            
        freq_Y = defaultdict(int)
        for y in Y_known:
            freq_Y[y] += 1
            
        R_min = 0
        keys = set(freq_T.keys()) | set(freq_Y.keys())
        for k in keys:
            cnt_T = freq_T.get(k, 0)
            cnt_Y = freq_Y.get(k, 0)
            R_min += max(0, cnt_T - cnt_Y)
            
        if R_min <= c_B:
            print("Yes")
        else:
            print("No")
            
    else:
        if len(X_known) <= c_B:
            print("Yes")
            return
            
        L = 0
        if X_known:
            L = max(L, max(X_known))
        if Y_known:
            L = max(L, max(Y_known))
            
        candidates = set()
        candidates.add(L)
        for a in X_known:
            for b in Y_known:
                s0 = a + b
                if s0 >= L:
                    candidates.add(s0)
                    
        found = False
        for S in candidates:
            valid = True
            for i in range(n):
                if A[i] != -1 and B[i] == -1:
                    if S < A[i]:
                        valid = False
                        break
                if A[i] == -1 and B[i] != -1:
                    if S < B[i]:
                        valid = False
                        break
            if not valid:
                continue
                
            T_list = [S - a for a in X_known]
            freq_T = defaultdict(int)
            for x in T_list:
                freq_T[x] += 1
                
            freq_Y = defaultdict(int)
            for y in Y_known:
                freq_Y[y] += 1
                
            R_min = 0
            keys = set(freq_T.keys()) | set(freq_Y.keys())
            for k in keys:
                cnt_T = freq_T.get(k, 0)
                cnt_Y = freq_Y.get(k, 0)
                R_min += max(0, cnt_T - cnt_Y)
                
            if R_min <= c_B:
                found = True
                break
                
        print("Yes" if found else "No")

if __name__ == "__main__":
    main()