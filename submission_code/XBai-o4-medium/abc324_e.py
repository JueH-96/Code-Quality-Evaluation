import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = input[ptr]
    ptr += 1
    m = len(T)
    pre_list = []
    suf_list = []
    
    for _ in range(N):
        S = input[ptr]
        ptr += 1
        # Compute prefix
        t_ptr = 0
        for c in S:
            if t_ptr < m and c == T[t_ptr]:
                t_ptr += 1
        pre = t_ptr
        # Compute suffix
        k = 0
        for c in S:
            if k < m and c == T[m - 1 - k]:
                k += 1
        suf = k
        pre_list.append(pre)
        suf_list.append(suf)
    
    # Sort the suffix list for binary search
    suf_list.sort()
    
    ans = 0
    for pre in pre_list:
        target = max(0, m - pre)
        idx = bisect.bisect_left(suf_list, target)
        ans += len(suf_list) - idx
    
    print(ans)

if __name__ == "__main__":
    main()