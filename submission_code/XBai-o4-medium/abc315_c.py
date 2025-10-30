import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    flavor_map = defaultdict(list)
    for _ in range(N):
        F_i = int(input[ptr])
        S_i = int(input[ptr + 1])
        ptr += 2
        flavor_map[F_i].append(S_i)
    
    max_same = -float('inf')
    max_per_flavor = []
    
    for f in flavor_map:
        s_list = flavor_map[f]
        s_list.sort(reverse=True)
        if len(s_list) >= 2:
            current = s_list[0] + s_list[1] / 2
            if current > max_same:
                max_same = current
        max_per_flavor.append(s_list[0])
    
    max_per_flavor.sort(reverse=True)
    if len(max_per_flavor) >= 2:
        max_diff = max_per_flavor[0] + max_per_flavor[1]
    else:
        max_diff = -float('inf')
    
    final_ans = max(max_same, max_diff)
    print(int(final_ans))

if __name__ == "__main__":
    main()