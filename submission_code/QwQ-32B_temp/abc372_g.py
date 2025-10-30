import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx]); idx +=1
    for _ in range(T):
        N = int(input[idx]); idx +=1
        A = []
        B = []
        C = []
        valid = True
        for _ in range(N):
            a = int(input[idx]); idx +=1
            b = int(input[idx]); idx +=1
            c = int(input[idx]); idx +=1
            A.append(a)
            B.append(b)
            C.append(c)
            if a + b >= c:
                valid = False
        if not valid:
            print(0)
            continue
        # Compute x_max
        x_max = float('inf')
        for i in range(N):
            numerator = C[i] - B[i] - 1
            denominator = A[i]
            if denominator <= 0:
                valid = False
                break
            current = numerator // denominator
            if current < 0:
                valid = False
                break
            if current < x_max:
                x_max = current
        if not valid or x_max < 1:
            print(0)
            continue
        # Compute the minimum for each x from 1 to x_max
        total = 0
        for x in range(1, x_max + 1):
            min_val = float('inf')
            for i in range(N):
                val = (C[i] - A[i] * x - 1) // B[i]
                if val < min_val:
                    min_val = val
            if min_val >= 1:
                total += min_val
        print(total)
        
if __name__ == "__main__":
    main()