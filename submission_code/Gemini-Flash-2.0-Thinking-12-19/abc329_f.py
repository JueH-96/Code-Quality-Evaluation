def solve():
    n, q = map(int, input().split())
    initial_colors = list(map(int, input().split()))
    box_colors = []
    for color in initial_colors:
        box_colors.append({color})
    
    for _ in range(q):
        a, b = map(int, input().split())
        source_box_index = a - 1
        destination_box_index = b - 1
        source_colors = box_colors[source_box_index]
        destination_colors = box_colors[destination_box_index]
        updated_destination_colors = destination_colors.union(source_colors)
        box_colors[destination_box_index] = updated_destination_colors
        box_colors[source_box_index] = set()
        print(len(box_colors[destination_box_index]))

if __name__ == '__main__':
    solve()