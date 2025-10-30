import itertools
import math

def main():
    grid = []
    for _ in range(3):
        data = input().split()
        grid.extend(map(int, data))
    
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    
    sensitive_lines = []
    for triple in lines:
        a, b, c = triple
        x, y, z = grid[a], grid[b], grid[c]
        if (x == y and x != z) or (x == z and x != y) or (y == z and y != x):
            sensitive_lines.append(triple)
            
    total_permutations = math.factorial(9)
    good_count = 0
    for p in itertools.permutations(range(9)):
        time_arr = [0] * 9
        for idx, cell in enumerate(p):
            time_arr[cell] = idx
            
        bad_found = False
        for triple in sensitive_lines:
            a, b, c = triple
            cells = [a, b, c]
            sorted_cells = sorted(cells, key=lambda x: time_arr[x])
            n0 = grid[sorted_cells[0]]
            n1 = grid[sorted_cells[1]]
            n2 = grid[sorted_cells[2]]
            if n0 == n1 and n0 != n2:
                bad_found = True
                break
                
        if not bad_found:
            good_count += 1
            
    probability = good_count / total_permutations
    print("{:.20f}".format(probability))

if __name__ == "__main__":
    main()