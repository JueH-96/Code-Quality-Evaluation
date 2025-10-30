import sys

class BoxData:
    __slots__ = ('color_count', 'size')
    def __init__(self):
        self.color_count = dict()
        self.size = 0

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    boxes = [None] * (N + 1)  # boxes[1..N]
    for i in range(1, N+1):
        bd = BoxData()
        color = C[i-1]
        bd.color_count[color] = 1
        bd.size = 1
        boxes[i] = bd
    
    output = []
    for _ in range(Q):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        
        data_a = boxes[a]
        data_b = boxes[b]
        
        if data_a.size == 0:
            output.append(str(data_b.size))
            continue
        
        if data_b.size == 0:
            # merge a into b by swapping after merge
            pass
        else:
            if data_a.size > data_b.size:
                # Merge b into a, then swap a and b
                while data_b.color_count:
                    color, cnt = data_b.color_count.popitem()
                    if color in data_a.color_count:
                        data_a.color_count[color] += cnt
                    else:
                        data_a.color_count[color] = cnt
                        data_a.size += 1
                data_b.size = 0
                # Swap
                boxes[a], boxes[b] = boxes[b], boxes[a]
            else:
                # Merge a into b
                while data_a.color_count:
                    color, cnt = data_a.color_count.popitem()
                    if color in data_b.color_count:
                        data_b.color_count[color] += cnt
                    else:
                        data_b.color_count[color] = cnt
                        data_b.size += 1
                data_a.size = 0
        
        output.append(str(boxes[b].size))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()