import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = int(input[idx])
    T = int(input[idx+1])
    idx += 2
    
    segments = []
    for _ in range(N):
        A = int(input[idx])
        B = int(input[idx+1])
        C = int(input[idx+2])
        D = int(input[idx+3])
        idx += 4
        x1, y1 = A, B
        x2, y2 = C, D
        length = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
        segments.append(( (x1, y1), (x2, y2), length ))
    
    min_time = float('inf')
    
    for perm in itertools.permutations(segments):
        for directions in itertools.product([0, 1], repeat=len(perm)):
            current_x, current_y = 0.0, 0.0
            total_time = 0.0
            for i in range(len(perm)):
                seg = perm[i]
                if directions[i] == 0:
                    start = seg[0]
                    end = seg[1]
                else:
                    start = seg[1]
                    end = seg[0]
                
                dx = start[0] - current_x
                dy = start[1] - current_y
                distance = (dx**2 + dy**2) ** 0.5
                move_time = distance / S
                total_time += move_time
                total_time += seg[2] / T
                current_x, current_y = end
            
            if total_time < min_time:
                min_time = total_time
    
    print("{0:.20f}".format(min_time))

if __name__ == "__main__":
    main()