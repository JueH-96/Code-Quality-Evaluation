import sys

def solve():
    line1 = sys.stdin.readline().split()
    a = int(line1[0])
    b = int(line1[1])
    c = int(line1[2])
    d = int(line1[3])
    e = int(line1[4])
    f = int(line1[5])
    line2 = sys.stdin.readline().split()
    g = int(line2[0])
    h = int(line2[1])
    i = int(line2[2])
    j = int(line2[3])
    k = int(line2[4])
    l = int(line2[5])
    
    x_start = max(a, g)
    x_end = min(d, j)
    y_start = max(b, h)
    y_end = min(e, k)
    z_start = max(c, i)
    z_end = min(f, l)
    
    if x_end > x_start and y_end > y_start and z_end > z_start:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()