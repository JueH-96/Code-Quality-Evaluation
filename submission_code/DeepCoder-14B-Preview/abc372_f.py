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
        for _ in range(N):
            a = int(input[ptr])
            b = int(input[ptr+1])
            c = int(input[ptr+2])
            A.append(a)
            B.append(b)
            C.append(c)
            ptr += 3
        
        x_max_list = [(c - 1) // a for a, c in zip(A, C)]
        x_max = min(x_max_list)
        if x_max < 1:
            print(0)
            continue
        
        x_min_y1_list = [(c - b - 1) // a for a, b, c in zip(A, B, C)]
        x_min_y1 = min(x_min_y1_list)
        if x_min_y1 < 1:
            print(0)
            continue
        
        x_max_to_process = min(x_max, x_min_y1)
        total = 0
        for x in range(1, x_max_to_process + 1):
            min_y = float('inf')
            valid = True
            for i in range(N):
                numerator = C[i] - A[i] * x - 1
                if numerator < 0:
                    valid = False
                    break
                y_i = numerator // B[i]
                if y_i < min_y:
                    min_y = y_i
            if not valid:
                continue
            if min_y >= 1:
                total += min_y
        print(total)

if __name__ == '__main__':
    main()