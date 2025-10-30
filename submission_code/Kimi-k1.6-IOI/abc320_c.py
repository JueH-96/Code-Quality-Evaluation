def main():
    import sys
    input = sys.stdin.read().split()
    M = int(input[0])
    S1 = input[1]
    S2 = input[2]
    S3 = input[3]
    
    min_time = float('inf')
    
    for d in '0123456789':
        if d not in S1 or d not in S2 or d not in S3:
            continue
        
        residues = []
        for s in [S1, S2, S3]:
            res = [i for i, char in enumerate(s) if char == d]
            residues.append(res)
        
        r1_list, r2_list, r3_list = residues
        
        for r1 in r1_list:
            for r2 in r2_list:
                for r3 in r3_list:
                    times1 = [r1, r1 + M, r1 + 2 * M]
                    times2 = [r2, r2 + M, r2 + 2 * M]
                    times3 = [r3, r3 + M, r3 + 2 * M]
                    
                    for t1 in times1:
                        for t2 in times2:
                            for t3 in times3:
                                if t1 != t2 and t2 != t3 and t1 != t3:
                                    current_max = max(t1, t2, t3)
                                    if current_max < min_time:
                                        min_time = current_max
    
    if min_time != float('inf'):
        print(min_time)
    else:
        print(-1)

if __name__ == "__main__":
    main()