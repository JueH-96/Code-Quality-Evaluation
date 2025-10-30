from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    M = int(input[idx])
    idx +=1
    S = input[idx]
    idx +=1
    T = input[idx]
    idx +=1
    
    if M > N:
        print("No")
        return
    
    # Precompute windows for each position
    windows = [[] for _ in range(N)]
    for i in range(N):
        s_start = max(0, i - M + 1)
        s_end = min(i, N - M)
        for s in range(s_start, s_end + 1):
            windows[i].append(s)
    
    visited = [False] * (N - M + 1)
    stamped = [False] * N
    q = deque()
    
    # Initialize queue with all valid stamps
    for s in range(N - M + 1):
        match = True
        for j in range(M):
            if S[s + j] != T[j]:
                match = False
                break
        if match:
            visited[s] = True
            q.append(s)
    
    while q:
        s = q.popleft()
        # Mark all positions in s..s+M-1 as stamped
        for i in range(s, s + M):
            stamped[i] = True
        
        # Check all windows that include any of the positions s..s+M-1
        for i in range(s, s + M):
            for s_prime in windows[i]:
                if not visited[s_prime]:
                    # Check if all positions in s_prime's window are valid
                    valid = True
                    for j in range(M):
                        pos = s_prime + j
                        if pos >= N:
                            valid = False
                            break
                        if S[pos] != T[j] and not stamped[pos]:
                            valid = False
                            break
                    if valid:
                        visited[s_prime] = True
                        q.append(s_prime)
    
    # Check if all positions are stamped
    all_stamped = all(stamped)
    print("Yes" if all_stamped else "No")

if __name__ == "__main__":
    main()