import sys

def main():
    data = sys.stdin.read().splitlines()
    if len(data) < 12:
        print("No")
        return
        
    polys = []
    for i in range(3):
        p = []
        for j in range(4):
            line = data[i*4 + j].strip()
            p.append(line)
        polys.append(p)
    
    total_hash = 0
    for p in polys:
        for row in p:
            total_hash += row.count('#')
            
    if total_hash != 16:
        print("No")
        return
        
    placements = []
    
    for poly in polys:
        base_set = set()
        for i in range(4):
            for j in range(4):
                if poly[i][j] == '#':
                    base_set.add((i, j))
                    
        rotations = [base_set]
        current = base_set
        for _ in range(3):
            next_rot = set()
            for (i, j) in current:
                next_rot.add((j, 3-i))
            rotations.append(next_rot)
            current = next_rot
            
        seen = set()
        poly_placements = []
        
        for rot in rotations:
            if not rot:
                continue
            min_i = min(i for (i, j) in rot)
            max_i = max(i for (i, j) in rot)
            min_j = min(j for (i, j) in rot)
            max_j = max(j for (i, j) in rot)
            
            dx_min = -min_i
            dx_max = 3 - max_i
            dy_min = -min_j
            dy_max = 3 - max_j
            
            for dx in range(dx_min, dx_max + 1):
                for dy in range(dy_min, dy_max + 1):
                    new_set = set()
                    valid = True
                    for (i, j) in rot:
                        ni = i + dx
                        nj = j + dy
                        if ni < 0 or ni >= 4 or nj < 0 or nj >= 4:
                            valid = False
                            break
                        new_set.add((ni, nj))
                    if not valid:
                        continue
                    f_set = frozenset(new_set)
                    if f_set not in seen:
                        seen.add(f_set)
                        poly_placements.append(new_set)
                        
        placements.append(poly_placements)
        
    found = False
    for p0 in placements[0]:
        for p1 in placements[1]:
            if p0 & p1:
                continue
            for p2 in placements[2]:
                if not (p0 & p2) and not (p1 & p2):
                    union_set = p0 | p1 | p2
                    if len(union_set) == 16:
                        found = True
                        break
            if found:
                break
        if found:
            break
            
    print("Yes" if found else "No")

if __name__ == '__main__':
    main()