import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    cuboids = []
    for _ in range(N):
        x1 = int(input[idx])
        y1 = int(input[idx+1])
        z1 = int(input[idx+2])
        x2 = int(input[idx+3])
        y2 = int(input[idx+4])
        z2 = int(input[idx+5])
        cuboids.append((x1, y1, z1, x2, y2, z2))
        idx +=6

    # Preprocess dictionaries
    start_x = defaultdict(list)
    end_x = defaultdict(list)
    start_y = defaultdict(list)
    end_y = defaultdict(list)
    start_z = defaultdict(list)
    end_z = defaultdict(list)
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        start_x[x1].append(i)
        end_x[x2].append(i)
        start_y[y1].append(i)
        end_y[y2].append(i)
        start_z[z1].append(i)
        end_z[z2].append(i)

    result = []
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        count = 0

        # Check x-axis
        # start_x[x2]
        for j in start_x.get(x2, []):
            if j == i:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = cuboids[j]
            # Check y and z overlap
            y_overlap = max(y1, jy1) < min(y2, jy2)
            z_overlap = max(z1, jz1) < min(z2, jz2)
            if y_overlap and z_overlap:
                count +=1
                break
        # end_x[x1]
        for j in end_x.get(x1, []):
            if j == i:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = cuboids[j]
            y_overlap = max(y1, jy1) < min(y2, jy2)
            z_overlap = max(z1, jz1) < min(z2, jz2)
            if y_overlap and z_overlap:
                count +=1
                break

        # Check y-axis
        # start_y[y2]
        for j in start_y.get(y2, []):
            if j == i:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = cuboids[j]
            x_overlap = max(x1, jx1) < min(x2, jx2)
            z_overlap = max(z1, jz1) < min(z2, jz2)
            if x_overlap and z_overlap:
                count +=1
                break
        # end_y[y1]
        for j in end_y.get(y1, []):
            if j == i:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = cuboids[j]
            x_overlap = max(x1, jx1) < min(x2, jx2)
            z_overlap = max(z1, jz1) < min(z2, jz2)
            if x_overlap and z_overlap:
                count +=1
                break

        # Check z-axis
        # start_z[z2]
        for j in start_z.get(z2, []):
            if j == i:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = cuboids[j]
            x_overlap = max(x1, jx1) < min(x2, jx2)
            y_overlap = max(y1, jy1) < min(y2, jy2)
            if x_overlap and y_overlap:
                count +=1
                break
        # end_z[z1]
        for j in end_z.get(z1, []):
            if j == i:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = cuboids[j]
            x_overlap = max(x1, jx1) < min(x2, jx2)
            y_overlap = max(y1, jy1) < min(y2, jy2)
            if x_overlap and y_overlap:
                count +=1
                break

        result.append(count)

    print('
'.join(map(str, result)))

if __name__ == "__main__":
    main()