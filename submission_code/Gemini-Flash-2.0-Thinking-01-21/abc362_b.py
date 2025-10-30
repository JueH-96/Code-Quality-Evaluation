import sys

def solve():
    lines = sys.stdin.readlines()
    coords = []
    for line in lines:
        coords.append(list(map(int, line.strip().split())))
    
    x_a, y_a = coords[0]
    x_b, y_b = coords[1]
    x_c, y_c = coords[2]
    
    ab_sq = (x_b - x_a)**2 + (y_b - y_a)**2
    bc_sq = (x_c - x_b)**2 + (y_c - y_b)**2
    ac_sq = (x_c - x_a)**2 + (y_c - y_a)**2
    
    if bc_sq + ac_sq == ab_sq or bc_sq + ab_sq == ac_sq or ac_sq + ab_sq == bc_sq:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()