import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    
    prev = {}
    next_dict = {}
    head = A[0]
    
    for i in range(N):
        a = A[i]
        if i == 0:
            prev[a] = None
        else:
            prev[a] = A[i-1]
        if i == N-1:
            next_dict[a] = None
        else:
            next_dict[a] = A[i+1]
    
    while ptr < len(input):
        while ptr < len(input) and input[ptr] == '':
            ptr += 1
        if ptr >= len(input):
            break
        if input[ptr] == '1':
            ptr += 1
            x = int(input[ptr])
            ptr += 1
            y = int(input[ptr])
            ptr += 1
            next_x = next_dict[x]
            next_dict[x] = y
            prev[y] = x
            next_dict[y] = next_x
            if next_x is not None:
                prev[next_x] = y
        elif input[ptr] == '2':
            ptr += 1
            x = int(input[ptr])
            ptr += 1
            prev_x = prev[x]
            next_x = next_dict[x]
            if prev_x is not None:
                next_dict[prev_x] = next_x
            if next_x is not None:
                prev[next_x] = prev_x
            if head == x:
                head = next_x
        else:
            ptr += 1
    
    res = []
    current = head
    while current is not None:
        res.append(str(current))
        current = next_dict.get(current, None)
    print(' '.join(res))

if __name__ == "__main__":
    main()