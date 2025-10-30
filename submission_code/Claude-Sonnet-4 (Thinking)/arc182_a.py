N, Q = map(int, input().split())
operations = []
for _ in range(Q):
    p, v = map(int, input().split())
    operations.append((p, v))

from functools import lru_cache

@lru_cache(None)
def dp(i, state):
    if i == Q:
        return 1
    
    p, v = operations[i]
    count = 0
    
    # Choice 1: affect range [0, p) (0-indexed)
    valid = True
    new_state = list(state)
    for j in range(p):
        if state[j] > v:
            valid = False
            break
        new_state[j] = v
    if valid:
        count = (count + dp(i + 1, tuple(new_state))) % 998244353
    
    # Choice 2: affect range [p-1, N) (0-indexed)  
    valid = True
    new_state = list(state)
    for j in range(p - 1, N):
        if state[j] > v:
            valid = False
            break
        new_state[j] = v
    if valid:
        count = (count + dp(i + 1, tuple(new_state))) % 998244353
    
    return count

initial_state = tuple([0] * N)
print(dp(0, initial_state))