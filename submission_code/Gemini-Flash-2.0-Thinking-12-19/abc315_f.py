import math

def solve():
    n = int(input())
    coords = []
    for _ in range(n):
        x, y = map(int, input().split())
        coords.append((x, y))
    
    def get_distance(p1_index, p2_index):
        x1, y1 = coords[p1_index-1]
        x2, y2 = coords[p2_index-1]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
        
    dp = {}
    dp[(1, 0)] = 0.0
    
    for i in range(1, n + 1):
        for skipped_count in range(n - 1):
            if (i, skipped_count) not in dp:
                continue
            current_min_cost = dp[(i, skipped_count)]
            for next_checkpoint in range(i + 1, n + 1):
                skipped_in_step = next_checkpoint - i - 1
                next_skipped_count = skipped_count + skipped_in_step
                if next_skipped_count < n - 1:
                    distance = get_distance(i, next_checkpoint)
                    new_cost = current_min_cost + distance
                    if (next_checkpoint, next_skipped_count) not in dp or new_cost < dp[(next_checkpoint, next_skipped_count)]:
                        dp[(next_checkpoint, next_skipped_count)] = new_cost
                        
    min_final_cost = float('inf')
    for skipped_count in range(n - 1):
        if (n, skipped_count) in dp:
            distance_cost = dp[(n, skipped_count)]
            penalty = 0
            if skipped_count > 0:
                penalty = 2**(skipped_count - 1)
            total_cost = distance_cost + penalty
            min_final_cost = min(min_final_cost, total_cost)
            
    print(f"{min_final_cost:.12f}")

if __name__ == '__main__':
    solve()