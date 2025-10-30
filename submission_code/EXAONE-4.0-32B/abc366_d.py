import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    N = int(next(it))
    arr = [[[0] * N for _ in range(N)] for __ in range(N)]
    
    for x in range(N):
        for y in range(N):
            for z in range(N):
                arr[x][y][z] = int(next(it))
    
    pref = [[[0] * (N+1) for _ in range(N+1)] for __ in range(N+1)]
    
    for i in range(1, N+1):
        for j in range(1, N+1):
            for k in range(1, N+1):
                val = arr[i-1][j-1][k-1]
                pref[i][j][k] = val + pref[i-1][j][k] + pref[i][j-1][k] + pref[i][j][k-1] \
                                - pref[i-1][j-1][k] - pref[i-1][j][k-1] - pref[i][j-1][k-1] \
                                + pref[i-1][j-1][k-1]
    
    Q = int(next(it))
    output_lines = []
    for _ in range(Q):
        Lx = int(next(it)); Rx = int(next(it))
        Ly = int(next(it)); Ry = int(next(it))
        Lz = int(next(it)); Rz = int(next(it))
        
        x_low = Lx - 1
        x_high = Rx
        y_low = Ly - 1
        y_high = Ry
        z_low = Lz - 1
        z_high = Rz
        
        total = pref[x_high][y_high][z_high] \
                - pref[x_low][y_high][z_high] \
                - pref[x_high][y_low][z_high] \
                - pref[x_high][y_high][z_low] \
                + pref[x_low][y_low][z_high] \
                + pref[x_low][y_high][z_low] \
                + pref[x_high][y_low][z_low] \
                - pref[x_low][y_low][z_low]
        
        output_lines.append(str(total))
    
    print("
".join(output_lines))

if __name__ == '__main__':
    main()