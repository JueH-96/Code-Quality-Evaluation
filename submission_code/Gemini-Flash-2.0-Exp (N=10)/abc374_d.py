import math

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve():
    n, s, t = map(int, input().split())
    segments = []
    for _ in range(n):
        segments.append(list(map(int, input().split())))

    min_time = float('inf')

    import itertools
    for perm in itertools.permutations(range(n)):
        for start_bits in range(2**n):
            current_time = 0.0
            current_x, current_y = 0, 0

            for i in range(n):
                seg_idx = perm[i]
                a, b, c, d = segments[seg_idx]
                
                if (start_bits >> i) & 1:
                    next_x, next_y = a, b
                    end_x, end_y = c, d
                else:
                    next_x, next_y = c, d
                    end_x, end_y = a, b

                current_time += dist(current_x, current_y, next_x, next_y) / s
                current_time += dist(next_x, next_y, end_x, end_y) / t
                current_x, current_y = end_x, end_y
            
            min_time = min(min_time, current_time)
    
    print(min_time)

solve()