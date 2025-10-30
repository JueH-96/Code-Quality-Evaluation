import bisect
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
        A = input[ptr]
        ptr += 1
        B = input[ptr]
        ptr += 1
        
        S = []
        T_base = []
        for i in range(N):
            if A[i] == '1':
                S.append(i + 1)
        for i in range(N):
            if B[i] == '1':
                T_base.append(i + 1)
        
        sum_A = len(S)
        sum_B = len(T_base)
        
        if sum_B == 0 or sum_A < sum_B:
            results.append("-1")
            continue
        
        m = sum_B
        base = sum_A // m
        extra = sum_A % m
        
        # Build prefix array
        prefix = [0] * (m + 1)
        current = 0
        for j in range(m):
            if j < extra:
                cnt = base + 1
            else:
                cnt = base
            current += cnt
            prefix[j + 1] = current
        
        max_diff = 0
        for k in range(sum_A):
            pos = bisect.bisect_right(prefix, k) - 1
            tbj = T_base[pos]
            diff = abs(S[k] - tbj)
            if diff > max_diff:
                max_diff = diff
        
        results.append(str(max_diff))
    
    print("
".join(results))

if __name__ == "__main__":
    main()