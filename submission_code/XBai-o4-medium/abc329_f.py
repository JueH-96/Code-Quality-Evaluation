import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    boxes = [set() for _ in range(N)]
    for i in range(N):
        boxes[i].add(C[i])
    
    for _ in range(Q):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        set_a = boxes[a-1]
        set_b = boxes[b-1]
        
        if len(set_a) > len(set_b):
            set_a.update(set_b)
            boxes[b-1] = set_a
            boxes[a-1] = set()
        else:
            set_b.update(set_a)
            set_a.clear()
        
        print(len(boxes[b-1]))

if __name__ == "__main__":
    main()