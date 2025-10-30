import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pos_map = defaultdict(list)
    for idx, val in enumerate(A):
        pos_map[val].append(idx)
    
    ans = 0
    for val in pos_map:
        pos_list = pos_map[val]
        m = len(pos_list)
        if m < 2:
            continue
        
        first_sum = 0
        second_sum = 0
        for i in range(m):
            p_i = pos_list[i]
            diff = p_i - i
            first_sum += diff * i
            second_sum += diff * (m - i - 1)
        
        ans += (first_sum - second_sum)
    
    print(ans)

if __name__ == "__main__":
    main()