import sys

def main():
    M = int(sys.stdin.readline())
    S1 = sys.stdin.readline().strip()
    S2 = sys.stdin.readline().strip()
    S3 = sys.stdin.readline().strip()
    
    min_time = float('inf')
    
    for d in map(str, range(10)):
        reel1 = [i for i in range(M) if S1[i] == d]
        reel2 = [i for i in range(M) if S2[i] == d]
        reel3 = [i for i in range(M) if S3[i] == d]
        
        if not reel1 or not reel2 or not reel3:
            continue
        
        for a in reel1:
            for b in reel2:
                for c in reel3:
                    current_min = float('inf')
                    for x in range(3):
                        t1 = a + x * M
                        for y in range(3):
                            t2 = b + y * M
                            for z in range(3):
                                t3 = c + z * M
                                if len({t1, t2, t3}) == 3:
                                    current_max = max(t1, t2, t3)
                                    if current_max < current_min:
                                        current_min = current_max
                    if current_min < min_time:
                        min_time = current_min
    
    print(-1 if min_time == float('inf') else min_time)

if __name__ == "__main__":
    main()