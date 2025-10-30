import bisect
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("0 0")
        return
    
    W, H = map(int, data[0].split())
    N = int(data[1])
    strawberries = []
    for i in range(2, 2 + N):
        p, q = map(int, data[i].split())
        strawberries.append((p, q))
    
    A = int(data[2 + N])
    cuts_x = list(map(int, data[2 + N + 1].split()))
    B = int(data[2 + N + 2])
    cuts_y = list(map(int, data[2 + N + 3].split()))
    
    total_pieces = (A + 1) * (B + 1)
    count_dict = {}
    
    for p, q in strawberries:
        idx_x = bisect.bisect_left(cuts_x, p)
        idx_y = bisect.bisect_left(cuts_y, q)
        key = (idx_x, idx_y)
        count_dict[key] = count_dict.get(key, 0) + 1
        
    distinct_pieces = len(count_dict)
    if distinct_pieces < total_pieces:
        min_val = 0
    else:
        min_val = min(count_dict.values())
        
    max_val = max(count_dict.values()) if N > 0 else 0
    
    print(f"{min_val} {max_val}")

if __name__ == "__main__":
    main()