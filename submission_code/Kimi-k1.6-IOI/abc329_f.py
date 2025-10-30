class Box:
    __slots__ = ['color_map', 'total_balls']
    def __init__(self, color):
        self.color_map = {color: 1}
        self.total_balls = 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    C = list(map(int, data[idx:idx+N]))
    idx += N
    
    boxes = []
    for color in C:
        box = Box(color)
        boxes.append(box)
    
    for _ in range(Q):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        box_a = boxes[a-1]
        box_b = boxes[b-1]
        
        # Ensure box_a has smaller or equal number of colors
        if len(box_a.color_map) > len(box_b.color_map):
            # Swap their color_maps and total_balls
            box_a.color_map, box_b.color_map = box_b.color_map, box_a.color_map
            box_a.total_balls, box_b.total_balls = box_b.total_balls, box_a.total_balls
        
        # Merge box_a into box_b
        for color, count in box_a.color_map.items():
            if color in box_b.color_map:
                box_b.color_map[color] += count
            else:
                box_b.color_map[color] = count
        
        box_b.total_balls += box_a.total_balls
        
        # Clear box_a's data
        box_a.color_map.clear()
        box_a.total_balls = 0
        
        # Output the number of distinct colors in box_b
        print(len(box_b.color_map))

if __name__ == "__main__":
    main()