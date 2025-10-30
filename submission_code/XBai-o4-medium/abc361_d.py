import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    if sorted(S) != sorted(T):
        print(-1)
        return
    
    initial = S + '..'
    target_str = T + '..'
    
    if initial == target_str:
        print(0)
        return
    
    visited = set()
    q = deque()
    q.append((initial, 0))
    visited.add(initial)
    
    while q:
        current, steps = q.popleft()
        empty_pos = -1
        for i in range(len(current) - 1):
            if current[i] == '.' and current[i+1] == '.':
                empty_pos = i
                break
        
        for x in range(len(current) - 1):
            if current[x] != '.' and current[x+1] != '.':
                new_state = list(current)
                c1 = new_state[x]
                c2 = new_state[x+1]
                new_state[empty_pos] = c1
                new_state[empty_pos + 1] = c2
                new_state[x] = '.'
                new_state[x+1] = '.'
                new_str = ''.join(new_state)
                if new_str == target_str:
                    print(steps + 1)
                    return
                if new_str not in visited:
                    visited.add(new_str)
                    q.append((new_str, steps + 1))
    
    print(-1)

if __name__ == '__main__':
    main()