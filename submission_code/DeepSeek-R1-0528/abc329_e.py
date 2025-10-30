import collections
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, m = map(int, data[0].split())
    S = data[1].strip()
    T = data[2].strip()
    
    state = list(S)
    q = collections.deque()
    total_segments = n - m + 1
    if total_segments <= 0:
        if all(c == '#' for c in state):
            print("Yes")
        else:
            print("No")
        return
            
    enqueued = [False] * total_segments
    
    for i in range(total_segments):
        all_sharp = True
        valid = True
        for j in range(m):
            if state[i+j] != '#':
                all_sharp = False
                if state[i+j] != T[j]:
                    valid = False
                    break
        if all_sharp:
            continue
        if valid:
            q.append(i)
            enqueued[i] = True
            
    while q:
        i = q.popleft()
        enqueued[i] = False
        
        all_sharp = True
        valid = True
        for j in range(m):
            if state[i+j] != '#':
                all_sharp = False
                if state[i+j] != T[j]:
                    valid = False
                    break
        if all_sharp:
            continue
        if not valid:
            continue
            
        changed = False
        for j in range(m):
            if state[i+j] != '#':
                state[i+j] = '#'
                changed = True
                
        if not changed:
            continue
            
        low_bound = max(0, i - m + 1)
        high_bound = min(i + m, total_segments)
        for k in range(low_bound, high_bound):
            if not enqueued[k]:
                enqueued[k] = True
                q.append(k)
                
    if all(c == '#' for c in state):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()