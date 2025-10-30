from collections import deque
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    N = int(data[0].strip())
    S = data[1].strip()
    T = data[2].strip()
    
    if sorted(S) != sorted(T):
        print(-1)
        return
        
    initial = S + ".."
    target = T + ".."
    
    if initial == target:
        print(0)
        return
        
    dist = {}
    dist[initial] = 0
    queue = deque([initial])
    
    while queue:
        state = queue.popleft()
        n = len(state)
        gap_index = None
        for i in range(n-1):
            if state[i] == '.' and state[i+1] == '.':
                gap_index = i
                break
        if gap_index is None:
            continue
            
        for j in range(n-1):
            if state[j] != '.' and state[j+1] != '.':
                lst = list(state)
                a, b = lst[j], lst[j+1]
                lst[gap_index] = a
                lst[gap_index+1] = b
                lst[j] = '.'
                lst[j+1] = '.'
                new_state = ''.join(lst)
                
                if new_state not in dist:
                    dist[new_state] = dist[state] + 1
                    if new_state == target:
                        print(dist[new_state])
                        return
                    queue.append(new_state)
                    
    print(-1)

if __name__ == "__main__":
    main()