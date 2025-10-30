import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    flavors = defaultdict(list)
    for _ in range(N):
        F, S = map(int, sys.stdin.readline().split())
        flavors[F].append(S)
    
    same_max = -float('inf')
    for f in flavors:
        s_list = sorted(flavors[f], reverse=True)
        if len(s_list) >= 2:
            candidate = s_list[0] + s_list[1] // 2
            if candidate > same_max:
                same_max = candidate
    
    max_s_list = [max(s_list) for s_list in flavors.values()]
    max_s_list.sort(reverse=True)
    
    diff_max = -float('inf')
    if len(max_s_list) >= 2:
        diff_max = max_s_list[0] + max_s_list[1]
    
    ans = max(same_max, diff_max)
    print(ans)

if __name__ == "__main__":
    main()