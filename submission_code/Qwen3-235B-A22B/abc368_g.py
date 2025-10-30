import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    max_val = 10**18
    output = []
    
    for _ in range(Q):
        t = int(data[idx])
        idx += 1
        if t == 1 or t == 2:
            i = int(data[idx]) - 1
            idx += 1
            x = int(data[idx])
            idx += 1
            if t == 1:
                A[i] = x
            else:
                B[i] = x
        else:
            l = int(data[idx])
            r = int(data[idx+1])
            idx += 2
            v = 0
            # Process elements from l to r (1-based)
            for i in range(l-1, r):
                if v == max_val:
                    break
                a = A[i]
                b = B[i]
                if b == 1:
                    new_v = v + a
                else:
                    add_val = v + a
                    mul_val = v * b
                    new_v = add_val if add_val > mul_val else mul_val
                if new_v > max_val:
                    v = max_val
                else:
                    v = new_v
            output.append(str(v))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()