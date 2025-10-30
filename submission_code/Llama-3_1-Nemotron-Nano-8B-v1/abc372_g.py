import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = []
        B = []
        C = []
        x_max_list = []
        for _ in range(N):
            a = int(input[ptr])
            b = int(input[ptr+1])
            c = int(input[ptr+2])
            ptr += 3
            A.append(a)
            B.append(b)
            C.append(c)
            x_max_i = (c - b - 1) // a
            x_max_list.append(x_max_i)
        x_global_max = min(x_max_list)
        if x_global_max < 1:
            print(0)
            continue
        total = 0
        for x in range(1, x_global_max + 1):
            min_y = float('inf')
            for i in range(N):
                numerator = C[i] - A[i] * x - 1
                if numerator < 0:
                    continue
                y = numerator // B[i]
                if y < min_y:
                    min_y = y
            if min_y == float('inf'):
                break
            total += min_y
        print(total)

if __name__ == '__main__':
    main()