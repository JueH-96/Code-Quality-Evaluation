import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    S = [0] * (N + 1)
    for i in range(1, N+1):
        S[i] = (S[i-1] + A[i-1]) % M
    
    total_sum_mod = S[N]
    
    pos_dict = defaultdict(list)
    for i in range(N+1):
        pos_dict[S[i]].append(i)
    
    total = 0
    for i in range(N):
        r = S[i]
        list_r = pos_dict[r]
        pos = bisect.bisect_left(list_r, i)
        count1 = len(list_r) - pos - 1
        
        target = (r - total_sum_mod) % M
        list_t = pos_dict.get(target, [])
        pos_t = bisect.bisect_left(list_t, i)
        count2 = pos_t
        
        total += count1 + count2
    
    print(total)

if __name__ == "__main__":
    main()