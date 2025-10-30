import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve():
    n = int(input())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))

    dp = {}

    def calculate_min_distance(current_index, visited_mask):
        if current_index == n - 1:
            skipped_count = bin(visited_mask).count('0') - 1
            penalty = 2**(skipped_count - 1) if skipped_count > 0 else 0
            return penalty

        if (current_index, visited_mask) in dp:
            return dp[(current_index, visited_mask)]

        min_dist = float('inf')
        for next_index in range(current_index + 1, n):
            new_visited_mask = visited_mask
            if next_index != n-1:
                new_visited_mask |= (1 << next_index)
            
            
            current_dist = dist(points[current_index], points[next_index])
            
            
            min_dist = min(min_dist, current_dist + calculate_min_distance(next_index, new_visited_mask))

        dp[(current_index, visited_mask)] = min_dist
        return min_dist

    result = calculate_min_distance(0, 1)
    print(result)

solve()