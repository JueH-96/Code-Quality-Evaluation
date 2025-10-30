import random
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    
    R1 = [0] * (n + 1)
    R2 = [0] * (n + 1)
    random.seed()
    for i in range(1, n + 1):
        R1[i] = random.getrandbits(64)
        R2[i] = random.getrandbits(64)
    
    PA1 = [0] * (n + 1)
    PA2 = [0] * (n + 1)
    for i in range(1, n + 1):
        x = A[i - 1]
        PA1[i] = PA1[i - 1] + R1[x]
        PA2[i] = PA2[i - 1] + R2[x]
        
    PB1 = [0] * (n + 1)
    PB2 = [0] * (n + 1)
    for i in range(1, n + 1):
        x = B[i - 1]
        PB1[i] = PB1[i - 1] + R1[x]
        PB2[i] = PB2[i - 1] + R2[x]
    
    out_lines = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        L = int(next(it))
        R_val = int(next(it))
        lenA = r - l + 1
        lenB = R_val - L + 1
        if lenA != lenB:
            out_lines.append("No")
        else:
            hashA1 = PA1[r] - PA1[l - 1]
            hashA2 = PA2[r] - PA2[l - 1]
            hashB1 = PB1[R_val] - PB1[L - 1]
            hashB2 = PB2[R_val] - PB2[L - 1]
            if hashA1 == hashB1 and hashA2 == hashB2:
                out_lines.append("Yes")
            else:
                out_lines.append("No")
                
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()