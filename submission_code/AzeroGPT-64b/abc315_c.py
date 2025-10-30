from collections import defaultdict

def get_max_satisfaction(N, cups):
    max_sats = defaultdict(list)
    max_val = 0
    for flavor, sat in cups:
        max_sats[flavor].append(sat)
        max_val = max(max_val, sat)

    for sats in max_sats.values():
        sats.sort(reverse=True)
    
    total_sats = [0, 0]
    total_flavors = max_sats.keys()
    for flavor in total_flavors:
        for sat in max_sats[flavor]:
            if total_sats[0] + sat // 2 > total_sats[1]:
                total_sats[1] = total_sats[0] + sat // 2
            total_sats[0] += sat
    
    max_satisfaction = 0
    for flavor in total_flavors:
        first_sat = max_sats[flavor][0]
        second_sat = max_sats[flavor][1] if len(max_sats[flavor]) > 1 else 0
        
        total_sats_except_flavor = total_sats[0] - first_sat - second_sat
        satisfaction_with_same_flavors = total_sats_except_flavor + first_sat + second_sat // 2
        satisfaction_with_different_flavors = total_sats_except_flavor + first_sat + max_val if second_sat + max_val == first_sat * 2 else total_sats_except_flavor + first_sat + max(0, second_sat, max_val)
        
        max_satisfaction = max(max_satisfaction, satisfaction_with_same_flavors, satisfaction_with_different_flavors)
    
    return max_satisfaction

def ice_cream_max_satisfaction():
    N = int(input())
    cups = [list(map(int, input().split())) for _ in range(N)]
    print(get_max_satisfaction(N, cups))

ice_cream_max_satisfaction()