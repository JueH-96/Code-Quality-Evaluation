import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    cuboids = []
    index = 1
    for i in range(n):
        x1 = int(data[index]); y1 = int(data[index+1]); z1 = int(data[index+2])
        x2 = int(data[index+3]); y2 = int(data[index+4]); z2 = int(data[index+5])
        index += 6
        cuboids.append((x1, y1, z1, x2, y2, z2))
    
    events_x = defaultdict(lambda: {'start': [], 'end': []})
    events_y = defaultdict(lambda: {'start': [], 'end': []})
    events_z = defaultdict(lambda: {'start': [], 'end': []})
    
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
        events_x[x1]['start'].append((i, y1, y2, z1, z2))
        events_x[x2]['end'].append((i, y1, y2, z1, z2))
        events_y[y1]['start'].append((i, x1, x2, z1, z2))
        events_y[y2]['end'].append((i, x1, x2, z1, z2))
        events_z[z1]['start'].append((i, x1, x2, y1, y2))
        events_z[z2]['end'].append((i, x1, x2, y1, y2))
    
    ans = [0] * n
    
    for x_val in range(0, 101):
        if x_val not in events_x:
            continue
        E = events_x[x_val]['end']
        S = events_x[x_val]['start']
        for e in E:
            e_id, ey1, ey2, ez1, ez2 = e
            for s in S:
                s_id, sy1, sy2, sz1, sz2 = s
                if max(ey1, sy1) < min(ey2, sy2) and max(ez1, sz1) < min(ez2, sz2):
                    ans[e_id] += 1
                    ans[s_id] += 1
                    
    for y_val in range(0, 101):
        if y_val not in events_y:
            continue
        E = events_y[y_val]['end']
        S = events_y[y_val]['start']
        for e in E:
            e_id, ex1, ex2, ez1, ez2 = e
            for s in S:
                s_id, sx1, sx2, sz1, sz2 = s
                if max(ex1, sx1) < min(ex2, sx2) and max(ez1, sz1) < min(ez2, sz2):
                    ans[e_id] += 1
                    ans[s_id] += 1
                    
    for z_val in range(0, 101):
        if z_val not in events_z:
            continue
        E = events_z[z_val]['end']
        S = events_z[z_val]['start']
        for e in E:
            e_id, ex1, ex2, ey1, ey2 = e
            for s in S:
                s_id, sx1, sx2, sy1, sy2 = s
                if max(ex1, sx1) < min(ex2, sx2) and max(ey1, sy1) < min(ey2, sy2):
                    ans[e_id] += 1
                    ans[s_id] += 1
                    
    for res in ans:
        print(res)

if __name__ == '__main__':
    main()