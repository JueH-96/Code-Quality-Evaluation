import sys

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    idx = 1
    for _ in range(T):
        N = int(data[idx])
        X = int(data[idx+1])
        K = int(data[idx+2])
        idx += 3
        
        if K == 0:
            print(1)
            continue
        
        # Compute depth of X
        depth = 0
        tmp = X
        while tmp > 1:
            tmp >>= 1
            depth += 1
        
        # Handle a=0 case
        d = K
        left = X * (1 << d)  # 2^d using bit shift for efficiency
        if left > N:
            cnt = 0
        else:
            right = (X + 1) * (1 << d) - 1
            right = min(right, N)
            cnt = right - left + 1
            if cnt < 0:
                cnt = 0
        ans = cnt
        
        # Handle a from 1 to min(K-1, depth)
        max_a = min(K - 1, depth)
        for a in range(1, max_a + 1):
            current_d = K - a - 1
            if current_d < 0:
                continue
            A = X >> a
            if A == 0:
                continue
            # Compute C
            C = (X >> (a - 1)) ^ 1
            left_sub = C * (1 << current_d)
            if left_sub > N:
                continue
            right_sub = (C + 1) * (1 << current_d) - 1
            right_sub = min(right_sub, N)
            cnt_sub = right_sub - left_sub + 1
            if cnt_sub < 0:
                cnt_sub = 0
            ans += cnt_sub
        
        # Handle a=K case
        if K <= depth:
            A = X >> K
            if A != 0:
                ans += 1
        
        print(ans)

if __name__ == '__main__':
    main()