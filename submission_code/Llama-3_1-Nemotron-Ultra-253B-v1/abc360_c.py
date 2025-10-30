import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    W = list(map(int, input[ptr:ptr+N]))
    
    box_items = defaultdict(list)
    for a, w in zip(A, W):
        box_items[a].append(w)
    
    total = 0
    for items in box_items.values():
        if not items:
            continue
        sum_w = sum(items)
        max_w = max(items)
        total += sum_w - max_w
    print(total)

if __name__ == "__main__":
    main()