import sys
from collections import defaultdict

def compute_a_k(x):
    k = 0
    while x % 2 == 0:
        k += 1
        x = x // 2
    return (x, k)

class GroupData:
    def __init__(self):
        self.count_mod = [defaultdict(int) for _ in range(26)]  # m from 0 to 25
        self.sum_mod = [defaultdict(int) for _ in range(26)]

def process_group(group, a_j):
    sum_case2 = 0
    MAX_M = 25
    for m in range(MAX_M, 0, -1):
        mod = 1 << m
        r = (-a_j) % mod
        sum_a_r = group.sum_mod[m].get(r, 0)
        count_r = group.count_mod[m].get(r, 0)
        if m < MAX_M:
            higher_mod = 1 << (m + 1)
            r_higher = (-a_j) % higher_mod
            sum_a_higher = group.sum_mod[m+1].get(r_higher, 0)
            count_higher = group.count_mod[m+1].get(r_higher, 0)
            sum_a_r -= sum_a_higher
            count_r -= count_higher
        numerator = sum_a_r + a_j * count_r
        denominator = 1 << m
        sum_case2 += numerator // denominator
    
    # Update the group's data
    for m in range(1, MAX_M + 1):
        mod = 1 << m
        rem = a_j % mod
        group.count_mod[m][rem] += 1
        group.sum_mod[m][rem] += a_j
    
    return sum_case2

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    
    groups = dict()  # key: k, value: GroupData
    MAX_K = 30
    sum_a = [0] * (MAX_K + 2)
    count = [0] * (MAX_K + 2)
    total_sum = 0
    
    for x in A:
        a_j, k_j = compute_a_k(x)
        # Compute sum_case1a and sum_case1b
        sum_case1a = 0
        sum_a_less = 0
        sum_pow_less = 0
        for k in range(k_j):
            sum_a_less += sum_a[k]
            sum_pow_less += count[k] * (1 << (k_j - k))
        sum_case1a = sum_a_less + a_j * sum_pow_less
        
        sum_case1b = 0
        count_more = 0
        sum_a_pow_more = 0
        for k in range(k_j + 1, MAX_K + 1):
            count_more += count[k]
            sum_a_pow_more += sum_a[k] * (1 << (k - k_j))
        sum_case1b = a_j * count_more + sum_a_pow_more
        
        # Compute sum_case2
        if k_j in groups:
            group = groups[k_j]
        else:
            group = GroupData()
            groups[k_j] = group
        sum_case2 = process_group(group, a_j)
        
        # Add all contributions
        total_sum += sum_case1a + sum_case1b + sum_case2
        total_sum += a_j  # case i=j
        
        # Update sum_a and count
        sum_a[k_j] += a_j
        count[k_j] += 1
    
    print(total_sum)

if __name__ == "__main__":
    main()