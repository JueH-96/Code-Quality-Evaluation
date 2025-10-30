import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        points.append((x, y))
    
    group0 = []
    group1 = []
    
    for x, y in points:
        s = x + y
        d = x - y
        if s % 2 == 0:
            group0.append((s, d))
        else:
            group1.append((s, d))
    
    def calc_sum(arr):
        arr.sort()
        total = 0
        prefix = 0
        for i, num in enumerate(arr):
            total += i * num - prefix
            prefix += num
        return total

    total_ans = 0
    for group in [group0, group1]:
        if len(group) < 2:
            continue
        u_list = [a[0] for a in group]
        v_list = [a[1] for a in group]
        sum_u = calc_sum(u_list)
        sum_v = calc_sum(v_list)
        total_ans += (sum_u + sum_v) // 2
        
    print(total_ans)

if __name__ == "__main__":
    main()