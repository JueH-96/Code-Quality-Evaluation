import sys
import itertools
import math

def main():
    grid = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        grid.append(row)
    
    cell_values = [grid[i][j] for i in range(3) for j in range(3)]
    
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    
    line_vals = []
    for a, b, c in lines:
        va = cell_values[a]
        vb = cell_values[b]
        vc = cell_values[c]
        line_vals.append((va, vb, vc))
    
    count = 0
    total_perms = math.factorial(9)
    
    for perm in itertools.permutations(range(9)):
        pos = [0] * 9
        for idx, x in enumerate(perm):
            pos[x] = idx
        
        valid = True
        for i in range(len(lines)):
            a, b, c = lines[i]
            va, vb, vc = line_vals[i]
            pa = pos[a]
            pb = pos[b]
            pc = pos[c]
            
            if pa <= pb and pa <= pc:
                first_val = va
                if pb <= pc:
                    second_val = vb
                    third_val = vc
                else:
                    second_val = vc
                    third_val = vb
            elif pb <= pa and pb <= pc:
                first_val = vb
                if pa <= pc:
                    second_val = va
                    third_val = vc
                else:
                    second_val = vc
                    third_val = va
            else:
                first_val = vc
                if pa <= pb:
                    second_val = va
                    third_val = vb
                else:
                    second_val = vb
                    third_val = va
            
            if first_val == second_val and third_val != first_val:
                valid = False
                break
        
        if valid:
            count += 1
    
    probability = count / total_perms
    print("{0:.15f}".format(probability))

if __name__ == "__main__":
    main()