import sys

def solve():
    w, h = map(int, sys.stdin.readline().split())
    n = int(sys.stdin.readline())
    strawberries = []
    for _ in range(n):
        strawberries.append(tuple(map(int, sys.stdin.readline().split())))
    a = int(sys.stdin.readline())
    a_cuts = list(map(int, sys.stdin.readline().split())) if a > 0 else []
    b = int(sys.stdin.readline())
    b_cuts = list(map(int, sys.stdin.readline().split())) if b > 0 else []
    
    x_boundaries = [0] + sorted(a_cuts) + [w]
    y_boundaries = [0] + sorted(b_cuts) + [h]
    
    num_x_intervals = len(x_boundaries) - 1
    num_y_intervals = len(y_boundaries) - 1
    
    strawberry_counts = [[0] * num_y_intervals for _ in range(num_x_intervals)]
    
    for p, q in strawberries:
        x_interval_index = -1
        for i in range(1, len(x_boundaries)):
            if x_boundaries[i-1] <= p < x_boundaries[i]:
                x_interval_index = i - 1
                break
        y_interval_index = -1
        for j in range(1, len(y_boundaries)):
            if y_boundaries[j-1] <= q < y_boundaries[j]:
                y_interval_index = j - 1
                break
        if x_interval_index != -1 and y_interval_index != -1:
            strawberry_counts[x_interval_index][y_interval_index] += 1
            
    all_counts = []
    for i in range(num_x_intervals):
        for j in range(num_y_intervals):
            all_counts.append(strawberry_counts[i][j])
            
    min_strawberries = min(all_counts) if all_counts else 0
    max_strawberries = max(all_counts) if all_counts else 0
    
    print(min_strawberries, max_strawberries)

if __name__ == '__main__':
    solve()