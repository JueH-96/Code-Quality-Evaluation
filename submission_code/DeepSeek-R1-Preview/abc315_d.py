import sys
from collections import defaultdict

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    active = [[True for _ in range(W)] for _ in range(H)]
    
    rows = [{'active_count': W, 'color_set': defaultdict(int)} for _ in range(H)]
    cols = [{'active_count': H, 'color_set': defaultdict(int)} for _ in range(W)]
    
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            rows[i]['color_set'][c] += 1
            cols[j]['color_set'][c] += 1
    
    marked_rows = set()
    marked_cols = set()
    
    while True:
        rows_to_mark = []
        for i in range(H):
            if i not in marked_rows and rows[i]['active_count'] >= 2 and len(rows[i]['color_set']) == 1:
                rows_to_mark.append(i)
        
        cols_to_mark = []
        for j in range(W):
            if j not in marked_cols and cols[j]['active_count'] >= 2 and len(cols[j]['color_set']) == 1:
                cols_to_mark.append(j)
        
        if not rows_to_mark and not cols_to_mark:
            break
        
        marked_rows.update(rows_to_mark)
        marked_cols.update(cols_to_mark)
        
        to_remove = set()
        for i in rows_to_mark:
            for j in range(W):
                if active[i][j]:
                    to_remove.add((i, j))
        
        for j in cols_to_mark:
            for i in range(H):
                if active[i][j]:
                    to_remove.add((i, j))
        
        for i, j in to_remove:
            if active[i][j]:
                active[i][j] = False
                c = grid[i][j]
                
                rows[i]['active_count'] -= 1
                rows[i]['color_set'][c] -= 1
                if rows[i]['color_set'][c] == 0:
                    del rows[i]['color_set'][c]
                
                cols[j]['active_count'] -= 1
                cols[j]['color_set'][c] -= 1
                if cols[j]['color_set'][c] == 0:
                    del cols[j]['color_set'][c]
    
    count = 0
    for i in range(H):
        for j in range(W):
            if active[i][j]:
                count += 1
    print(count)

if __name__ == '__main__':
    main()