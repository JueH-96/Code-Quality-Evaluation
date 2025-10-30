import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    flavor_groups = defaultdict(list)
    for _ in range(N):
        F_i, S_i = map(int, sys.stdin.readline().split())
        flavor_groups[F_i].append(S_i)
    
    # Sort each group in descending order
    for fl in flavor_groups:
        flavor_groups[fl].sort(reverse=True)
    
    max_case2 = -float('inf')
    for fl in flavor_groups:
        group = flavor_groups[fl]
        if len(group) >= 2:
            s1 = group[0]
            s2 = group[1]
            current = s1 + (s2 // 2)
            if current > max_case2:
                max_case2 = current
    
    # Prepare max_list for case1
    max_list = []
    for fl in flavor_groups:
        max_list.append(flavor_groups[fl][0])
    
    # Sort max_list in descending order
    max_list.sort(reverse=True)
    case1_candidate = -float('inf')
    if len(max_list) >= 2:
        case1_candidate = max_list[0] + max_list[1]
    
    # Determine the answer
    ans = max(case1_candidate, max_case2)
    print(ans)

if __name__ == "__main__":
    main()