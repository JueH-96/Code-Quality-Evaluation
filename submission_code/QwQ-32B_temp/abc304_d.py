import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    W = int(input[idx])
    H = int(input[idx+1])
    idx += 2
    
    N = int(input[idx])
    idx +=1
    
    strawberries = []
    for _ in range(N):
        p = int(input[idx])
        q = int(input[idx+1])
        strawberries.append((p, q))
        idx +=2
    
    A = int(input[idx])
    idx +=1
    a = list(map(int, input[idx:idx+A]))
    idx +=A
    
    B = int(input[idx])
    idx +=1
    b = list(map(int, input[idx:idx+B]))
    idx +=B
    
    count = defaultdict(int)
    
    for p, q in strawberries:
        v = bisect.bisect_left(a, p)
        h = bisect.bisect_left(b, q)
        count[(v, h)] +=1
    
    total_cells = (A +1) * (B +1)
    current_occupied = len(count)
    
    if current_occupied < total_cells:
        min_val = 0
    else:
        min_val = min(count.values())
    
    if count:
        max_val = max(count.values())
    else:
        max_val =0  # This case shouldn't happen as N >=1
    
    print(min_val, max_val)

if __name__ == "__main__":
    main()