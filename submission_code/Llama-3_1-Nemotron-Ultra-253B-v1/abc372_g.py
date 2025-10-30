import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr +=1
    for _ in range(T):
        N = int(input[ptr])
        ptr +=1
        A = []
        B = []
        C = []
        valid = True
        for __ in range(N):
            a = int(input[ptr])
            b = int(input[ptr+1])
            c = int(input[ptr+2])
            ptr +=3
            A.append(a)
            B.append(b)
            C.append(c)
            if c <= b:
                valid = False
        if not valid:
            print(0)
            continue
        x_max = float('inf')
        for i in range(N):
            numerator = C[i] - B[i] -1
            if numerator <0:
                x_max = 0
                break
            xi = numerator // A[i]
            if xi < x_max:
                x_max = xi
        if x_max <1:
            print(0)
            continue
        current_x = 1
        sum_total =0
        while current_x <= x_max:
            current_m = float('inf')
            for i in range(N):
                term = (C[i] - A[i] * current_x -1) // B[i]
                if term < current_m:
                    current_m = term
            if current_m ==0:
                break
            next_x = x_max
            for i in range(N):
                numerator = C[i] -1 - B[i] * current_m
                if numerator <0:
                    temp = 0
                else:
                    temp = numerator // A[i]
                if temp < next_x:
                    next_x = temp
            next_x = min(next_x, x_max)
            if next_x < current_x:
                break
            sum_total += current_m * (next_x - current_x +1)
            current_x = next_x +1
        print(sum_total)

if __name__ == '__main__':
    main()