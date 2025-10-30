# YOUR CODE HERE
def solve():
    ha, wa = map(int, input().split())
    a = [input() for _ in range(ha)]
    hb, wb = map(int, input().split())
    b = [input() for _ in range(hb)]
    hx, wx = map(int, input().split())
    x = [input() for _ in range(hx)]

    a_blacks = []
    for r in range(ha):
        for c in range(wa):
            if a[r][c] == '#':
                a_blacks.append((r, c))

    b_blacks = []
    for r in range(hb):
        for c in range(wb):
            if b[r][c] == '#':
                b_blacks.append((r, c))

    x_blacks = []
    for r in range(hx):
        for c in range(wx):
            if x[r][c] == '#':
                x_blacks.append((r, c))
    
    if len(a_blacks) + len(b_blacks) != len(x_blacks):
        print("No")
        return

    for ar_offset in range(-ha + 1, hx):
        for ac_offset in range(-wa + 1, wx):
            for br_offset in range(-hb + 1, hx):
                for bc_offset in range(-wb + 1, wx):
                    
                    combined_blacks = []
                    for r, c in a_blacks:
                        nr, nc = r + ar_offset, c + ac_offset
                        if 0 <= nr < hx and 0 <= nc < wx:
                            combined_blacks.append((nr, nc))
                    
                    for r, c in b_blacks:
                        nr, nc = r + br_offset, c + bc_offset
                        if 0 <= nr < hx and 0 <= nc < wx:
                            combined_blacks.append((nr, nc))
                    
                    combined_blacks = sorted(list(set(combined_blacks)))
                    
                    if len(combined_blacks) != len(x_blacks):
                        continue
                    
                    match = True
                    for i in range(len(combined_blacks)):
                        if combined_blacks[i] != x_blacks[i]:
                            match = False
                            break
                    
                    if match:
                        print("Yes")
                        return
    
    print("No")

solve()