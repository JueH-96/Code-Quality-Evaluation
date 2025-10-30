def solve():
    m = int(input())
    s1 = input()
    s2 = input()
    s3 = input()
    min_overall_time = float('inf')
    possible = False
    
    for char_code in range(10):
        char = str(char_code)
        indices1 = []
        indices2 = []
        indices3 = []
        for i in range(m):
            if s1[i] == char:
                indices1.append(i)
            if s2[i] == char:
                indices2.append(i)
            if s3[i] == char:
                indices3.append(i)
        
        if not indices1 or not indices2 or not indices3:
            continue
            
        possible_for_char = False
        min_time_for_char = float('inf')
        
        for j1 in indices1:
            for j2 in indices2:
                for j3 in indices3:
                    times = sorted([j1, j2, j3])
                    t1, t2, t3 = times[0], times[1], times[2]
                    if t1 == t2 == t3:
                        t2 += m
                        t3 += 2 * m
                    elif t1 == t2:
                        t2 += m
                    elif t2 == t3:
                        t3 += m
                    current_time = max(t1, t2, t3)
                    min_time_for_char = min(min_time_for_char, current_time)
                    possible_for_char = True
                    
        if possible_for_char:
            min_overall_time = min(min_overall_time, min_time_for_char)
            possible = True
            
    if not possible:
        print("-1")
    else:
        print(min_overall_time)

if __name__ == '__main__':
    solve()