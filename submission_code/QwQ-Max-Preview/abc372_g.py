import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr +=1
        A = []
        B = []
        C = []
        for _ in range(N):
            a = int(input[ptr])
            b = int(input[ptr+1])
            c = int(input[ptr+2])
            ptr +=3
            A.append(a)
            B.append(b)
            C.append(c)
        
        # Compute x_max1 = min( (C_i-1) // A_i )
        x_max1 = (C[0]-1) // A[0]
        for i in range(1, N):
            if (C[i]-1) // A[i] < x_max1:
                x_max1 = (C[i]-1) // A[i]
        if x_max1 < 1:
            print(0)
            continue
        
        # Compute x_max2 = min( (C_i - B_i -1) // A_i )
        x_max2 = (C[0] - B[0] -1) // A[0]
        for i in range(1, N):
            temp = (C[i] - B[i] -1) // A[i]
            if temp < x_max2:
                x_max2 = temp
        x_max = min(x_max1, x_max2)
        if x_max < 1:
            print(0)
            continue
        
        # Now compute sum for x from 1 to x_max
        sum_ans = 0
        for x in range(1, x_max+1):
            min_y = float('inf')
            for i in range(N):
                numerator = C[i] - 1 - A[i] * x
                if numerator < 0:
                    min_y = -1
                    break
                y_i = numerator // B[i]
                if y_i < min_y:
                    min_y = y_i
            if min_y < 1:
                sum_ans += 0
            else:
                sum_ans += min_y
        print(sum_ans)

if __name__ == '__main__':
    main()