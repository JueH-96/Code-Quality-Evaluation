# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    total_sum = sum(a)
    max_min_w = 0
    uncut_lines = 0

    for min_w in range(1, total_sum // k + 1):
        possible_divisions = 0
        uncut_lines_for_min_w = 0
        
        for start in range(n):
            current_sum = 0
            num_people = 0
            current_uncut = 0
            valid_division = True
            
            for i in range(n):
                current_sum += a[(start + i) % n]
                if current_sum >= min_w:
                    num_people += 1
                    current_sum = 0
                    if num_people > k:
                        valid_division = False
                        break
            
            if valid_division and num_people == k and current_sum == 0:
                possible_divisions += 1
                
                current_uncut = 0
                cuts = [False] * n
                current_sum = 0
                person_count = 0
                current_start = start
                
                for i in range(n):
                    current_sum += a[(start + i) % n]
                    if current_sum >= min_w:
                        person_count += 1
                        current_sum = 0
                        for j in range(current_start, (start + i + 1) % n if (start + i + 1) % n != 0 else n):
                            cuts[j] = True
                        current_start = (start + i + 1) % n
                
                for i in range(n):
                    if not cuts[i]:
                        current_uncut += 1
                uncut_lines_for_min_w |= (1 << current_uncut)
        
        if possible_divisions > 0:
            max_min_w = min_w
            uncut_lines_temp = 0
            for i in range(n + 1):
                if (uncut_lines_for_min_w >> i) & 1:
                    uncut_lines_temp = i
            uncut_lines = uncut_lines_temp
    
    print(max_min_w, uncut_lines)

solve()