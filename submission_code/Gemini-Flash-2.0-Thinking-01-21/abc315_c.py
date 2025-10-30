import sys

def solve():
    n = int(sys.stdin.readline())
    cups = []
    for _ in range(n):
        f, s = map(int, sys.stdin.readline().split())
        cups.append({'flavor': f, 'deliciousness': s})
    
    max_diff_satisfaction = 0
    max_same_satisfaction = 0
    
    # Calculate max satisfaction for different flavors
    sorted_cups_diff = sorted(cups, key=lambda x: x['deliciousness'], reverse=True)
    if n >= 2:
        if sorted_cups_diff[0]['flavor'] != sorted_cups_diff[1]['flavor']:
            max_diff_satisfaction = sorted_cups_diff[0]['deliciousness'] + sorted_cups_diff[1]['deliciousness']
        else:
            found_diff_flavor = False
            for i in range(2, n):
                if sorted_cups_diff[i]['flavor'] != sorted_cups_diff[0]['flavor']:
                    max_diff_satisfaction = sorted_cups_diff[0]['deliciousness'] + sorted_cups_diff[i]['deliciousness']
                    found_diff_flavor = True
                    break
            if not found_diff_flavor:
                max_diff_satisfaction = 0
    else:
        max_diff_satisfaction = 0

    # Calculate max satisfaction for same flavors
    flavor_deliciousness = {}
    for cup in cups:
        flavor = cup['flavor']
        deliciousness = cup['deliciousness']
        if flavor not in flavor_deliciousness:
            flavor_deliciousness[flavor] = []
        flavor_deliciousness[flavor].append(deliciousness)
        
    max_same_flavor_satisfaction = 0
    for flavor in flavor_deliciousness:
        deliciousness_list = flavor_deliciousness[flavor]
        if len(deliciousness_list) >= 2:
            deliciousness_list.sort(reverse=True)
            s = deliciousness_list[0]
            t = deliciousness_list[1]
            satisfaction = s + t / 2
            max_same_flavor_satisfaction = max(max_same_flavor_satisfaction, satisfaction)
            
    print(int(max(max_diff_satisfaction, max_same_flavor_satisfaction)))

if __name__ == '__main__':
    solve()