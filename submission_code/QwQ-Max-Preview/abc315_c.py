import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    flavor_dict = defaultdict(list)
    for _ in range(N):
        F_i = int(input[idx])
        S_i = int(input[idx+1])
        idx +=2
        flavor_dict[F_i].append(S_i)
    
    # Process each flavor to keep top two elements
    for f in flavor_dict:
        s_list = flavor_dict[f]
        s_list.sort(reverse=True)
        if len(s_list) > 2:
            s_list = s_list[:2]
        flavor_dict[f] = s_list
    
    # Compute candidate2: max same-flavor satisfaction
    max_same = 0
    for s_list in flavor_dict.values():
        if len(s_list) >= 2:
            current = s_list[0] + (s_list[1] // 2)
            if current > max_same:
                max_same = current
    
    # Compute candidate1: max different-flavor satisfaction
    max_per_flavor = [s[0] for s in flavor_dict.values()]
    max_per_flavor.sort(reverse=True)
    candidate1 = 0
    if len(max_per_flavor) >= 2:
        candidate1 = max_per_flavor[0] + max_per_flavor[1]
    
    ans = max(candidate1, max_same)
    print(ans)

if __name__ == "__main__":
    main()