import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    W = int(data[ptr])
    H = int(data[ptr+1])
    ptr +=2
    N = int(data[ptr])
    ptr +=1
    strawberries = []
    for _ in range(N):
        p = int(data[ptr])
        q = int(data[ptr+1])
        strawberries.append((p, q))
        ptr +=2
    A = int(data[ptr])
    ptr +=1
    a_list = list(map(int, data[ptr:ptr+A]))
    ptr +=A
    B = int(data[ptr])
    ptr +=1
    b_list = list(map(int, data[ptr:ptr+B]))
    ptr +=B
    
    count_map = defaultdict(int)
    for p, q in strawberries:
        i = bisect.bisect_left(a_list, p)
        j = bisect.bisect_left(b_list, q)
        count_map[(i, j)] += 1
    
    max_count = max(count_map.values()) if count_map else 0
    total_pieces = (A + 1) * (B + 1)
    if total_pieces > len(count_map):
        min_count = 0
    else:
        min_count = min(count_map.values()) if count_map else 0
    
    print(min_count, max_count)

if __name__ == "__main__":
    main()