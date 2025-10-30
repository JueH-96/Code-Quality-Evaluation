from collections import defaultdict

def main():
    n = int(input())
    flavors = defaultdict(list)
    
    for _ in range(n):
        f, s = map(int, input().split())
        flavors[f].append(s)
    
    same_flavor_candidates = []
    max_per_flavor = []
    
    for f in flavors:
        s_list = sorted(flavors[f], reverse=True)
        if len(s_list) >= 2:
            same = s_list[0] + (s_list[1] // 2)
            same_flavor_candidates.append(same)
        if s_list:
            max_per_flavor.append(s_list[0])
    
    same_max = max(same_flavor_candidates) if same_flavor_candidates else 0
    
    max_per_flavor_sorted = sorted(max_per_flavor, reverse=True)
    diff_max = 0
    if len(max_per_flavor_sorted) >= 2:
        diff_max = max_per_flavor_sorted[0] + max_per_flavor_sorted[1]
    
    answer = max(same_max, diff_max)
    print(answer)

if __name__ == "__main__":
    main()