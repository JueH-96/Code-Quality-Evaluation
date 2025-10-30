def solve():
    w, h = map(int, input().split())
    n = int(input())
    strawberries = []
    for _ in range(n):
        strawberries.append(tuple(map(int, input().split())))
    a_count = int(input())
    a_lines = list(map(int, input().split())) if a_count > 0 else []
    b_count = int(input())
    b_lines = list(map(int, input().split())) if b_count > 0 else []
    
    x_boundaries = [0] + sorted(a_lines) + [w]
    y_boundaries = [0] + sorted(b_lines) + [h]
    
    piece_strawberry_counts = {}
    
    for p, q in strawberries:
        x_index = -1
        for i in range(1, len(x_boundaries)):
            if x_boundaries[i-1] < p < x_boundaries[i]:
                x_index = i
                break
        y_index = -1
        for j in range(1, len(y_boundaries)):
            if y_boundaries[j-1] < q < y_boundaries[j]:
                y_index = j
                break
        
        piece_index = (x_index, y_index)
        piece_strawberry_counts[piece_index] = piece_strawberry_counts.get(piece_index, 0) + 1
        
    min_strawberries = float('inf')
    max_strawberries = float('-inf')
    
    if not piece_strawberry_counts:
        min_strawberries = 0
        max_strawberries = 0
    else:
        for count in piece_strawberry_counts.values():
            min_strawberries = min(min_strawberries, count)
            max_strawberries = max(max_strawberries, count)
            
    if min_strawberries == float('inf'):
        min_strawberries = 0
    if max_strawberries == float('-inf'):
        max_strawberries = 0
        
    all_counts = []
    for i in range(1, len(x_boundaries)):
        for j in range(1, len(y_boundaries)):
            count = piece_strawberry_counts.get((i, j), 0)
            all_counts.append(count)
            
    if not all_counts:
        min_strawberries = 0
        max_strawberries = 0
    else:
        min_strawberries = min(all_counts)
        max_strawberries = max(all_counts)

    print(min_strawberries, max_strawberries)

if __name__ == '__main__':
    solve()