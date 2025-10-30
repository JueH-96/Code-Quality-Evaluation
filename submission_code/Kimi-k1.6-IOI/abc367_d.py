import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    mod_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        mod_sum[i] = (mod_sum[i-1] + A[i-1]) % M
    
    S_mod = mod_sum[N]
    
    remainders = defaultdict(list)
    for i in range(N + 1):
        remainders[mod_sum[i]].append(i)
    
    total = 0
    for s in range(1, N + 1):
        s_minus_1 = s - 1
        r1 = mod_sum[s_minus_1]
        # Calculate count1 using bisect for range [s+1, N]
        list_r1 = remainders.get(r1, [])
        left = bisect.bisect_left(list_r1, s + 1)
        right = bisect.bisect_right(list_r1, N)
        count1 = right - left
        
        # Calculate count2 using bisect for range [0, s-2]
        r2 = (r1 - S_mod) % M
        list_r2 = remainders.get(r2, [])
        count2 = bisect.bisect_right(list_r2, s - 2)
        
        total += count1 + count2
    
    print(total)

if __name__ == '__main__':
    main()