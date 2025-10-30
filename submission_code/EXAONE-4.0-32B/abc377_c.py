knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    m = int(data[1])
    pieces = []
    occ_set = set()
    idx = 2
    for i in range(m):
        a = int(data[idx])
        b = int(data[idx + 1])
        idx += 2
        pieces.append((a, b))
        occ_set.add((a, b))
    
    threatened = set()
    for (a, b) in pieces:
        for dx, dy in knight_moves:
            x = a + dx
            y = b + dy
            if 1 <= x <= n and 1 <= y <= n:
                threatened.add((x, y))
    
    common = threatened & occ_set
    num_common = len(common)
    num_threatened_empty = len(threatened) - num_common
    total_safe = n * n - m - num_threatened_empty
    print(total_safe)

if __name__ == '__main__':
    main()