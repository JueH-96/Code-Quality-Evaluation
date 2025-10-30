# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    flavor_dict = defaultdict(list)
    
    for _ in range(N):
        F, S = map(int, sys.stdin.readline().split())
        flavor_dict[F].append(S)
    
    # For different flavors
    max1 = -1
    max2 = -1
    for key in flavor_dict:
        current_max = max(flavor_dict[key])
        if current_max > max1:
            max2 = max1
            max1 = current_max
        elif current_max > max2:
            max2 = current_max
    
    if max1 != -1 and max2 != -1:
        diff_flavor_max = max1 + max2
    else:
        diff_flavor_max = -1
    
    # For same flavor
    same_flavor_max = -1
    for key in flavor_dict:
        if len(flavor_dict[key]) >= 2:
            sorted_S = sorted(flavor_dict[key], reverse=True)
            s = sorted_S[0]
            t = sorted_S[1]
            current = s + t // 2
            if current > same_flavor_max:
                same_flavor_max = current
    
    # Determine the overall maximum
    if diff_flavor_max == -1:
        print(same_flavor_max)
    elif same_flavor_max == -1:
        print(diff_flavor_max)
    else:
        print(max(diff_flavor_max, same_flavor_max))

if __name__ == "__main__":
    main()