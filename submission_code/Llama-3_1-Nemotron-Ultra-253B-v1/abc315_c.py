import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    flavor_dict = defaultdict(list)
    
    for _ in range(n):
        f, s = map(int, sys.stdin.readline().split())
        lst = flavor_dict[f]
        lst.append(s)
        lst.sort(reverse=True)
        if len(lst) > 2:
            lst.pop()
    
    same_flavor_max = 0
    for f in flavor_dict:
        lst = flavor_dict[f]
        if len(lst) >= 2:
            s1 = lst[0]
            s2 = lst[1]
            current = s1 + (s2 // 2)
            if current > same_flavor_max:
                same_flavor_max = current
    
    top_per_flavor = []
    for f in flavor_dict:
        lst = flavor_dict[f]
        if lst:
            top_per_flavor.append((lst[0], f))
    
    top_per_flavor.sort(reverse=True, key=lambda x: x[0])
    cross_flavor_max = 0
    
    if len(top_per_flavor) >= 2:
        first_s, first_f = top_per_flavor[0]
        for i in range(1, len(top_per_flavor)):
            s, f = top_per_flavor[i]
            if f != first_f:
                cross_flavor_max = first_s + s
                break
    
    print(max(same_flavor_max, cross_flavor_max))

if __name__ == "__main__":
    main()