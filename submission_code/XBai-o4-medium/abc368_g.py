def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    Q = int(input[ptr])
    ptr += 1
    
    for _ in range(Q):
        t = int(input[ptr])
        ptr += 1
        if t == 1:
            i = int(input[ptr]) - 1
            ptr += 1
            x = int(input[ptr])
            ptr += 1
            A[i] = x
        elif t == 2:
            i = int(input[ptr]) - 1
            ptr += 1
            x = int(input[ptr])
            ptr += 1
            B[i] = x
        else:
            l = int(input[ptr])
            ptr += 1
            r = int(input[ptr])
            ptr += 1
            start = l - 1
            end = r - 1
            current_add = A[start]
            current_mul = 0
            for i in range(start + 1, end + 1):
                prev_max = max(current_add, current_mul)
                current_add = prev_max + A[i]
                current_mul = prev_max * B[i]
            print(max(current_add, current_mul))

if __name__ == "__main__":
    main()