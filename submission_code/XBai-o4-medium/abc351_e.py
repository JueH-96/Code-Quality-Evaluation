import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    group_even = []
    group_odd = []
    
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr + 1])
        ptr += 2
        s = x + y
        u = s
        v = x - y
        if s % 2 == 0:
            group_even.append((u, v))
        else:
            group_odd.append((u, v))
    
    def compute_sum(arr):
        arr.sort()
        total = 0
        prefix = 0
        for i, x in enumerate(arr):
            total += x * i - prefix
            prefix += x
        return total
    
    total_sum = 0
    
    for group in [group_even, group_odd]:
        m = len(group)
        if m < 2:
            continue
        u_list = [u for u, v in group]
        v_list = [v for u, v in group]
        sum_u = compute_sum(u_list)
        sum_v = compute_sum(v_list)
        total_sum += (sum_u + sum_v) // 2
    
    print(total_sum)

if __name__ == "__main__":
    main()