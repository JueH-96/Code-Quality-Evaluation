import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pos_dict = defaultdict(list)
    for idx, val in enumerate(A):
        pos_dict[val].append(idx)
    
    total = 0
    
    for x in pos_dict:
        pos = pos_dict[x]
        m = len(pos)
        if m < 2:
            continue
        
        sum_p_a_a = 0
        sum_p_a_t = 0
        for a in range(m):
            sum_p_a_a += pos[a] * a
            sum_p_a_t += pos[a] * (m - a - 1)
        
        sum1 = sum_p_a_a - sum_p_a_t
        
        sum2 = 0
        for a in range(m - 1):
            t = m - a - 1
            sum2 += t * (t + 1) // 2
        
        total += sum1 - sum2
    
    print(total)

if __name__ == "__main__":
    main()