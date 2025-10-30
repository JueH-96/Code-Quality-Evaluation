def main():
    import sys
    input = sys.stdin.read().split()
    M = int(input[0])
    s1 = input[1]
    s2 = input[2]
    s3 = input[3]
    
    min_time = float('inf')
    
    for d in map(str, range(10)):
        if d not in s1 or d not in s2 or d not in s3:
            continue
        # Collect positions for each reel
        list1 = [i for i, c in enumerate(s1) if c == d]
        list2 = [i for i, c in enumerate(s2) if c == d]
        list3 = [i for i, c in enumerate(s3) if c == d]
        
        current_min = float('inf')
        for p1 in list1:
            for p2 in list2:
                for p3 in list3:
                    s = {p1, p2, p3}
                    if len(s) == 3:
                        t = max(p1, p2, p3)
                    elif len(s) == 2:
                        # Determine duplicated value
                        if p1 == p2:
                            dup = p1
                        elif p1 == p3:
                            dup = p1
                        else:
                            dup = p2
                        t = dup + M
                    else:  # len(s) == 1
                        t = p1 + 2 * M
                    if t < current_min:
                        current_min = t
        if current_min < min_time:
            min_time = current_min
    
    if min_time == float('inf'):
        print(-1)
    else:
        print(min_time)

if __name__ == '__main__':
    main()