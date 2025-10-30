def solve():
    n, q = map(int, input().split())
    colors = list(map(int, input().split()))
    boxes = []
    for color in colors:
        boxes.append({color})
    
    for _ in range(q):
        a, b = map(int, input().split())
        a_index = a - 1
        b_index = b - 1
        colors_to_move = boxes[a_index]
        boxes[b_index].update(colors_to_move)
        boxes[a_index] = set()
        print(len(boxes[b_index]))

if __name__ == '__main__':
    solve()