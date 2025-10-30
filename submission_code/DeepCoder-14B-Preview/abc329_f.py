def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    q = int(data[ptr])
    ptr += 1
    
    c = list(map(int, data[ptr:ptr + n]))
    ptr += n
    
    # Initialize boxes as a list of sets, with index 0 unused
    boxes = [set() for _ in range(n + 1)]
    for i in range(n):
        boxes[i + 1].add(c[i])
    
    output = []
    for _ in range(q):
        a = int(data[ptr])
        ptr += 1
        b = int(data[ptr])
        ptr += 1
        
        if boxes[a]:
            boxes[b].update(boxes[a])
            boxes[a].clear()
        
        output.append(str(len(boxes[b])))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()