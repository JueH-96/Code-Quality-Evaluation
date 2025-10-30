import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    cards = []
    idx = 1
    for i in range(n):
        a = int(data[idx])
        c = int(data[idx + 1])
        idx += 2
        cards.append((a, c, i + 1))
    
    cards.sort(key=lambda x: x[0])
    
    min_cost = float('inf')
    kept = []
    
    for i in range(n - 1, -1, -1):
        a_val, c_val, orig_idx = cards[i]
        if c_val < min_cost:
            min_cost = c_val
            kept.append(orig_idx)
            
    kept.sort()
    print(len(kept))
    if kept:
        print(" ".join(map(str, kept)))

if __name__ == "__main__":
    main()