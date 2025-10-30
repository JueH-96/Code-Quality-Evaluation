import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
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
            ptr +=3
            A.append(a)
            B.append(b)
            C.append(c)
        
        # Compute x_max
        x_max = float('inf')
        for i in range(N):
            if A[i] == 0:
                x_max = 0
                break
            x_i = (C[i] -1) // A[i]
            if x_i <1:
                x_max = 0
                break
            if x_i < x_max:
                x_max = x_i
        if x_max <1:
            results.append(0)
            continue
        
        # Compute K_max
        K_max = float('inf')
        for i in range(N):
            if B[i] ==0:
                K_i = 0
            else:
                numerator = (C[i] -1 - A[i])
                if numerator <0:
                    K_max = 0
                    break
                K_i = numerator // B[i]
            if K_i < K_max:
                K_max = K_i
        if K_max <1:
            results.append(0)
            continue
        
        # Now compute the sum
        S = 0
        for k in range(1, K_max +1):
            min_x = float('inf')
            for i in range(N):
                numerator = C[i] -1 - k * B[i]
                if numerator <0:
                    x_i = 0
                else:
                    x_i = numerator // A[i]
                if x_i < min_x:
                    min_x = x_i
            if min_x > x_max:
                min_x = x_max
            if min_x >=1:
                S += min_x
        results.append(S)
    
    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == '__main__':
    main()