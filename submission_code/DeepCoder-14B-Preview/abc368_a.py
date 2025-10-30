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
    output = []
    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            i = int(input[ptr]) - 1
            ptr += 1
            x = int(input[ptr])
            ptr += 1
            A[i] = x
        elif query == '2':
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
            v = 0
            for i in range(l-1, r):
                if i == l-1:
                    v += A[i]
                else:
                    add = v + A[i]
                    mult = v * B[i]
                    if mult > add:
                        v = mult
                    else:
                        v = add
            output.append(str(v))
    print('
'.join(output))

if __name__ == '__main__':
    main()