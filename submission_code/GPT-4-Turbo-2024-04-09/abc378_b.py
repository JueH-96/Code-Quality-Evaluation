import sys
input = sys.stdin.read

def solve():
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    
    schedules = []
    
    for _ in range(N):
        q = int(data[idx])
        r = int(data[idx + 1])
        schedules.append((q, r))
        idx += 2
    
    Q = int(data[idx])
    idx += 1
    
    results = []
    
    for _ in range(Q):
        t = int(data[idx]) - 1
        d = int(data[idx + 1])
        idx += 2
        
        q, r = schedules[t]
        
        # Calculate the next day the garbage of type t will be collected after day d
        if d % q == r:
            results.append(d)
        else:
            # Calculate the next collection day
            # If d % q > r, we need to add (q - (d % q - r))
            # If d % q < r, we need to add (r - d % q)
            if d % q > r:
                next_day = d + (q - (d % q - r))
            else:
                next_day = d + (r - d % q)
            results.append(next_day)
    
    for result in results:
        print(result)