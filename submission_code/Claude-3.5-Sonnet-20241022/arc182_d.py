from collections import deque

def is_good_sequence(seq, n):
    for i in range(n-1):
        if seq[i] == seq[i+1]:
            return False
    return True

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    # If sequences are already equal
    if a == b:
        return 0
    
    # BFS to find minimum operations
    visited = set()
    q = deque([(tuple(a), 0)])
    visited.add(tuple(a))
    
    while q:
        curr_seq, ops = q.popleft()
        curr_seq = list(curr_seq)
        
        # Try all possible operations
        for i in range(n):
            # Try +1
            curr_seq[i] = (curr_seq[i] + 1) % m
            if is_good_sequence(curr_seq, n):
                curr_tuple = tuple(curr_seq)
                if curr_tuple == tuple(b):
                    return ops + 1
                if curr_tuple not in visited:
                    visited.add(curr_tuple)
                    q.append((curr_tuple, ops + 1))
            curr_seq[i] = (curr_seq[i] - 1) % m
            
            # Try -1
            curr_seq[i] = (curr_seq[i] - 1) % m
            if is_good_sequence(curr_seq, n):
                curr_tuple = tuple(curr_seq)
                if curr_tuple == tuple(b):
                    return ops + 1
                if curr_tuple not in visited:
                    visited.add(curr_tuple)
                    q.append((curr_tuple, ops + 1))
            curr_seq[i] = (curr_seq[i] + 1) % m
    
    return -1

print(solve())