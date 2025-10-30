import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    cuboids = []
    index = 1
    for i in range(n):
        x1 = int(data[index])
        y1 = int(data[index+1])
        z1 = int(data[index+2])
        x2 = int(data[index+3])
        y2 = int(data[index+4])
        z2 = int(data[index+5])
        index += 6
        cuboids.append((i, x1, y1, z1, x2, y2, z2))
    
    ans = [0] * n
    
    L_x = [[] for _ in range(102)]
    R_x = [[] for _ in range(102)]
    L_y = [[] for _ in range(102)]
    R_y = [[] for _ in range(102)]
    L_z = [[] for _ in range(102)]
    R_z = [[] for _ in range(102)]
    
    for (id0, x1, y1, z1, x2, y2, z2) in cuboids:
        if x1 < x2:
            if x1 < 102:
                L_x[x1].append((id0, y1, y2, z1, z2))
            if x2 < 102:
                R_x[x2].append((id0, y1, y2, z1, z2))
        if y1 < y2:
            if y1 < 102:
                L_y[y1].append((id0, x1, x2, z1, z2))
            if y2 < 102:
                R_y[y2].append((id0, x1, x2, z1, z2))
        if z1 < z2:
            if z1 < 102:
                L_z[z1].append((id0, x1, x2, y1, y2))
            if z2 < 102:
                R_z[z2].append((id0, x1, x2, y1, y2))
                
    for coord in range(0, 102):
        L_list = L_x[coord]
        R_list = R_x[coord]
        if not L_list and not R_list:
            continue
        for cubR in R_list:
            idR = cubR[0]
            y1R = cubR[1]
            y2R = cubR[2]
            z1R = cubR[3]
            z2R = cubR[4]
            for cubL in L_list:
                idL = cubL[0]
                if idR == idL:
                    continue
                y1L = cubL[1]
                y2L = cubL[2]
                z1L = cubL[3]
                z2L = cubL[4]
                y_overlap_low = max(y1R, y1L)
                y_overlap_high = min(y2R, y2L)
                if y_overlap_high <= y_overlap_low:
                    continue
                z_overlap_low = max(z1R, z1L)
                z_overlap_high = min(z2R, z2L)
                if z_overlap_high <= z_overlap_low:
                    continue
                ans[idR] += 1
                ans[idL] += 1
                    
    for coord in range(0, 102):
        L_list = L_y[coord]
        R_list = R_y[coord]
        if not L_list and not R_list:
            continue
        for cubR in R_list:
            idR = cubR[0]
            x1R = cubR[1]
            x2R = cubR[2]
            z1R = cubR[3]
            z2R = cubR[4]
            for cubL in L_list:
                idL = cubL[0]
                if idR == idL:
                    continue
                x1L = cubL[1]
                x2L = cubL[2]
                z1L = cubL[3]
                z2L = cubL[4]
                x_overlap_low = max(x1R, x1L)
                x_overlap_high = min(x2R, x2L)
                if x_overlap_high <= x_overlap_low:
                    continue
                z_overlap_low = max(z1R, z1L)
                z_overlap_high = min(z2R, z2L)
                if z_overlap_high <= z_overlap_low:
                    continue
                ans[idR] += 1
                ans[idL] += 1
                    
    for coord in range(0, 102):
        L_list = L_z[coord]
        R_list = R_z[coord]
        if not L_list and not R_list:
            continue
        for cubR in R_list:
            idR = cubR[0]
            x1R = cubR[1]
            x2R = cubR[2]
            y1R = cubR[3]
            y2R = cubR[4]
            for cubL in L_list:
                idL = cubL[0]
                if idR == idL:
                    continue
                x1L = cubL[1]
                x2L = cubL[2]
                y1L = cubL[3]
                y2L = cubL[4]
                x_overlap_low = max(x1R, x1L)
                x_overlap_high = min(x2R, x2L)
                if x_overlap_high <= x_overlap_low:
                    continue
                y_overlap_low = max(y1R, y1L)
                y_overlap_high = min(y2R, y2L)
                if y_overlap_high <= y_overlap_low:
                    continue
                ans[idR] += 1
                ans[idL] += 1
                    
    for a in ans:
        print(a)

if __name__ == "__main__":
    main()