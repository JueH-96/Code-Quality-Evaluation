import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    groups = defaultdict(list)
    for x in a:
        k = 0
        temp = x
        while temp % 2 == 0:
            temp //= 2
            k += 1
        groups[k].append(temp)
    
    count_dict = {}
    sum_b_dict = {}
    for k in groups:
        count_dict[k] = len(groups[k])
        sum_b_dict[k] = sum(groups[k])
    
    total = 0
    keys = sorted(groups.keys())
    num_keys = len(keys)
    for i in range(num_keys):
        k1 = keys[i]
        for j in range(i+1, num_keys):
            k2 = keys[j]
            diff = k2 - k1
            total += count_dict[k2] * sum_b_dict[k1]
            total += (1 << diff) * count_dict[k1] * sum_b_dict[k2]
            
    for k in keys:
        lst = groups[k]
        n_group = len(lst)
        if n_group == 0:
            continue
        if n_group <= 1000:
            group_sum = 0
            for i in range(n_group):
                for j in range(i, n_group):
                    s_val = lst[i] + lst[j]
                    while s_val % 2 == 0:
                        s_val //= 2
                    group_sum += s_val
            total += group_sum
        else:
            group_sum = 0
            for i in range(n_group):
                for j in range(i, n_group):
                    s_val = lst[i] + lst[j]
                    while s_val % 2 == 0:
                        s_val //= 2
                    group_sum += s_val
            total += group_sum
            
    print(total)

if __name__ == '__main__':
    main()