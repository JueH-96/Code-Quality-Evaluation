import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    P = []
    R = []
    O = []
    
    for _ in range(N):
        T = int(input[ptr])
        X = int(input[ptr + 1])
        ptr += 2
        if T == 0:
            P.append(X)
        elif T == 1:
            R.append(X)
        else:
            O.append(X)
    
    # Sort in descending order
    P.sort(reverse=True)
    R.sort(reverse=True)
    O.sort(reverse=True)
    
    # Compute prefix sums
    sum_p = [0] * (len(P) + 1)
    for i in range(1, len(P) + 1):
        sum_p[i] = sum_p[i-1] + P[i-1]
    
    sum_r = [0] * (len(R) + 1)
    for i in range(1, len(R) + 1):
        sum_r[i] = sum_r[i-1] + R[i-1]
    
    sum_o = [0] * (len(O) + 1)
    for i in range(1, len(O) + 1):
        sum_o[i] = sum_o[i-1] + O[i-1]
    
    max_total = 0
    max_k = min(M, len(O))
    
    for k in range(0, max_k + 1):
        sum_X = sum_o[k]
        rem = M - k
        if rem < 0:
            continue
        total_non_opener = len(P) + len(R)
        if rem > total_non_opener:
            continue
        
        r_low = max(0, rem - len(P))
        r_high = min(len(R), rem, sum_X)
        if r_low > r_high:
            continue
        
        left = r_low
        right = r_high
        current_max = -1
        
        # Ternary search with a final brute force for small ranges
        while right - left > 5:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            val1 = sum_r[mid1] + sum_p[rem - mid1]
            val2 = sum_r[mid2] + sum_p[rem - mid2]
            if val1 < val2:
                left = mid1
            else:
                right = mid2
        
        # Check all values in the small remaining range
        for r in range(left, right + 1):
            if r < r_low or r > r_high:
                continue
            current = sum_r[r] + sum_p[rem - r]
            if current > current_max:
                current_max = current
        
        if current_max > max_total:
            max_total = current_max
    
    print(max_total)

if __name__ == '__main__':
    main()