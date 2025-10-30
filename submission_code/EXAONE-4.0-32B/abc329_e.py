import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    
    n, m = map(int, data[0].split())
    S = data[1].strip()
    T = data[2].strip()
    
    if not set(S).issubset(set(T)):
        print("No")
        return
        
    arr = list(S)
    if m == 0:
        print("Yes")
        return
        
    q = deque()
    in_queue = [False] * (n)
    
    for i in range(0, n - m + 1):
        flag = True
        for j in range(m):
            if arr[i+j] != '#' and arr[i+j] != T[j]:
                flag = False
                break
        if flag:
            q.append(i)
            in_queue[i] = True
            
    while q:
        i = q.popleft()
        in_queue[i] = False
        
        flag = True
        for j in range(m):
            if arr[i+j] != '#' and arr[i+j] != T[j]:
                flag = False
                break
                
        if not flag:
            continue
            
        changed = False
        for j in range(m):
            if arr[i+j] != '#':
                arr[i+j] = '#'
                changed = True
                
        if not changed:
            continue
            
        L = max(0, i - m + 1)
        R = min(n - m, i + m - 1)
        for k in range(L, R + 1):
            if in_queue[k]:
                continue
            flag_seg = True
            for j in range(m):
                if arr[k+j] != '#' and arr[k+j] != T[j]:
                    flag_seg = False
                    break
            if flag_seg:
                q.append(k)
                in_queue[k] = True
                
    if all(c == '#' for c in arr):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()