def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    a = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    prev = {}
    next_ = {}
    
    if N == 1:
        prev[a[0]] = None
        next_[a[0]] = None
    else:
        prev[a[0]] = None
        next_[a[0]] = a[1]
        for i in range(1, N-1):
            prev[a[i]] = a[i-1]
            next_[a[i]] = a[i+1]
        prev[a[-1]] = a[-2]
        next_[a[-1]] = None
    
    head = a[0]
    tail = a[-1]
    
    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        query_type = input[ptr]
        if query_type == '1':
            x = int(input[ptr+1])
            y = int(input[ptr+2])
            ptr +=3
            original_next_x = next_.get(x)
            next_[x] = y
            prev[y] = x
            next_[y] = original_next_x
            if original_next_x is not None:
                prev[original_next_x] = y
            else:
                tail = y
        else:
            x = int(input[ptr+1])
            ptr +=2
            p = prev.get(x)
            n = next_.get(x)
            if p is not None:
                next_[p] = n
            else:
                head = n
            if n is not None:
                prev[n] = p
            else:
                tail = p
            del prev[x]
            del next_[x]
    
    current = head
    result = []
    while current is not None:
        result.append(str(current))
        current = next_.get(current)
    print(' '.join(result))

if __name__ == "__main__":
    main()