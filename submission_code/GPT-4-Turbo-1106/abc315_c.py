import sys
from collections import defaultdict

def read_int():
    return int(sys.stdin.readline().strip())

def read_ints():
    return list(map(int, sys.stdin.readline().strip().split()))

def main():
    N = read_int()
    ice_creams = [read_ints() for _ in range(N)]
    
    # Group ice creams by flavor
    flavor_groups = defaultdict(list)
    for flavor, deliciousness in ice_creams:
        flavor_groups[flavor].append(deliciousness)
    
    # Find the top two deliciousness values for each flavor
    top_deliciousness = []
    for flavor, deliciousness_list in flavor_groups.items():
        deliciousness_list.sort(reverse=True)
        top_deliciousness.append(deliciousness_list[0])
        if len(deliciousness_list) > 1:
            top_deliciousness.append(deliciousness_list[1] // 2)  # Half the value of the second one
    
    # Find the top two deliciousness values overall
    top_deliciousness.sort(reverse=True)
    
    # Calculate the maximum satisfaction
    max_satisfaction = sum(top_deliciousness[:2])
    
    print(max_satisfaction)

if __name__ == "__main__":
    main()