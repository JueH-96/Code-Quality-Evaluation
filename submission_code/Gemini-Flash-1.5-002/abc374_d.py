import itertools
import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve():
    n, s, t = map(int, input().split())
    segments = []
    for _ in range(n):
        segments.append(list(map(int, input().split())))

    points = [(0, 0)]
    for a, b, c, d in segments:
        points.append((a, b))
        points.append((c, d))

    min_time = float('inf')
    for perm in itertools.permutations(range(1, 2 * n + 1)):
        cur_time = 0
        cur_pos = (0, 0)
        
        visited_segments = [False] * n
        
        for i in perm:
            seg_index = (i - 1) // 2
            point_index = (i - 1) % 2
            
            next_pos = (segments[seg_index][point_index * 2], segments[seg_index][point_index * 2 + 1])
            
            cur_time += dist(cur_pos, next_pos) / s
            cur_pos = next_pos
            
            next_pos = (segments[seg_index][(point_index + 1) * 2], segments[seg_index][(point_index + 1) * 2 + 1])
            cur_time += dist(cur_pos, next_pos) / t
            cur_pos = next_pos

        min_time = min(min_time, cur_time)

    print(min_time)

solve()