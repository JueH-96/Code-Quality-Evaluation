from collections import deque
import sys

def main():
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    n = len(S)
    m = len(T)
    
    # If the length difference is already more than K, impossible
    if abs(n - m) > K:
        print("No")
        return
    
    target_diff = n - m
    visited = {}
    q = deque()
    q.append((0, 0, 0))
    visited[(0, 0)] = 0
    found = False
    
    while q:
        i, j, s = q.popleft()
        
        if i == n and j == m:
            found = True
            break
        
        if s >= K:
            continue
        
        # Generate substitution state
        if i < n and j < m:
            new_i = i + 1
            new_j = j + 1
            new_s = s + (0 if S[i] == T[j] else 1)
            if new_s > K:
                continue
            new_d = new_i - new_j
            required = abs(target_diff - new_d)
            if new_s + required <= K:
                key = (new_i, new_j)
                if key not in visited or new_s < visited[key]:
                    visited[key] = new_s
                    q.append((new_i, new_j, new_s))
        
        # Generate deletion state
        new_i = i + 1
        new_j = j
        new_s = s + 1
        if new_i > n:
            pass
        elif new_s > K:
            pass
        else:
            new_d = new_i - new_j
            required = abs(target_diff - new_d)
            if new_s + required <= K:
                key = (new_i, new_j)
                if key not in visited or new_s < visited[key]:
                    visited[key] = new_s
                    q.append((new_i, new_j, new_s))
        
        # Generate insertion state
        new_i = i
        new_j = j + 1
        new_s = s + 1
        if new_j > m:
            pass
        elif new_s > K:
            pass
        else:
            new_d = new_i - new_j
            required = abs(target_diff - new_d)
            if new_s + required <= K:
                key = (new_i, new_j)
                if key not in visited or new_s < visited[key]:
                    visited[key] = new_s
                    q.append((new_i, new_j, new_s))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()