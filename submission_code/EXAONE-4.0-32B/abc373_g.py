import math
import sys

def solve(points):
    n = len(points) // 2
    if n == 0:
        return {}
    o = min(points, key=lambda p: (p[0], p[1]))
    rest = [p for p in points if p != o]
    
    rest_sorted = sorted(rest, key=lambda p: math.atan2(p[1]-o[1], p[0]-o[0]))
    
    balance = 0
    idx = None
    if o[2] == 'P':
        for i, p in enumerate(rest_sorted):
            if p[2] == 'P':
                balance += 1
            else:
                balance -= 1
            if balance == -1:
                idx = i
                break
    else:
        for i, p in enumerate(rest_sorted):
            if p[2] == 'P':
                balance += 1
            else:
                balance -= 1
            if balance == 1:
                idx = i
                break
                
    if idx is None:
        return {}
        
    last_point = rest_sorted[idx]
    match_here = {}
    if o[2] == 'P':
        match_here[o[3]] = last_point[3]
    else:
        match_here[last_point[3]] = o[3]
        
    group1 = [o] + rest_sorted[:idx+1]
    group1_recurse = [p for p in group1 if p != o and p != last_point]
    group2 = rest_sorted[idx+1:]
    
    match1 = solve(group1_recurse)
    match2 = solve(group2)
    
    match_here.update(match1)
    match_here.update(match2)
    return match_here

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    points = []
    index = 1
    for i in range(n):
        a, b = map(int, data[index].split())
        index += 1
        points.append((a, b, 'P', i+1))
    for j in range(n):
        c, d = map(int, data[index].split())
        index += 1
        points.append((c, d, 'Q', j+1))
        
    matching_dict = solve(points)
    
    if len(matching_dict) != n:
        print(-1)
    else:
        res = [0] * n
        for pid, qid in matching_dict.items():
            res[pid-1] = qid
        print(" ".join(map(str, res)))
        
if __name__ == "__main__":
    main()