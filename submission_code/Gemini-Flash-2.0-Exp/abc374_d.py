import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve():
    n, s, t = map(int, input().split())
    segments = []
    for _ in range(n):
        segments.append(list(map(int, input().split())))

    min_time = float('inf')

    import itertools
    for perm in itertools.permutations(range(n)):
        for start_bits in itertools.product([0, 1], repeat=n):
            current_time = 0.0
            current_x, current_y = 0, 0
            
            for i in range(n):
                segment_index = perm[i]
                a, b, c, d = segments[segment_index]
                start_bit = start_bits[segment_index]
                
                if start_bit == 0:
                    x1, y1, x2, y2 = a, b, c, d
                else:
                    x1, y1, x2, y2 = c, d, a, b
                
                current_time += distance(current_x, current_y, x1, y1) / s
                current_time += distance(x1, y1, x2, y2) / t
                current_x, current_y = x2, y2
            
            min_time = min(min_time, current_time)

    print(min_time)

solve()