import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    
    # Compute prefix_mod and total_mod
    prefix_mod = []
    current_sum = 0
    prefix_mod.append(current_sum % M)
    for i in range(N):
        current_sum += A[i]
        prefix_mod.append(current_sum % M)
    total_mod = prefix_mod[-1]
    
    # Case 1: Count pairs (s, t) where t > s and same residue
    cnt_case1 = defaultdict(int)
    case1 = 0
    for i in range(N):
        cnt_case1[prefix_mod[i]] += 1
    for count in cnt_case1.values():
        if count >= 2:
            case1 += count * (count - 1) // 2
    
    # Case 2: Count pairs (s, t) where t < s and (pre[s] == (pre[t] + total_mod) mod M)
    cnt_case2 = defaultdict(int)
    case2 = 0
    for i in range(N):
        target = (prefix_mod[i] - total_mod) % M
        case2 += cnt_case2.get(target, 0)
        cnt_case2[prefix_mod[i]] += 1
    
    print(case1 + case2)

if __name__ == '__main__':
    main()