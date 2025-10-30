import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+2*n]))
    q = int(data[1+2*n])
    index = 1+2*n+1
    output_lines = []
    
    for _ in range(q):
        t = int(data[index]); index += 1
        if t == 1:
            pos = int(data[index]) - 1; index += 1
            x = int(data[index]); index += 1
            A[pos] = x
        elif t == 2:
            pos = int(data[index]) - 1; index += 1
            x = int(data[index]); index += 1
            B[pos] = x
        else:
            l = int(data[index]) - 1; index += 1
            r = int(data[index]) - 1; index += 1
            state = 0
            for j in range(l, r + 1):
                add_val = state + A[j]
                mul_val = state * B[j]
                state = add_val if add_val > mul_val else mul_val
            output_lines.append(str(state))
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()