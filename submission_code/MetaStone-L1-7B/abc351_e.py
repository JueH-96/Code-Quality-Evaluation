def compute_sum(group):
    if len(group) < 2:
        return 0
    
    a_list = [(x + y) // 2 for x, y in group]
    a_list.sort()
    n = len(a_list)
    sum_a = 0
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + a_list[i]
    for i in range(1, n):
        diff = prefix[i] - a_list[i] * i
        sum_a += abs(diff)
    
    b_list = [(x - y) // 2 for x, y in group]
    b_list.sort()
    sum_b = 0
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + b_list[i]
    for i in range(1, n):
        diff = prefix[i] - b_list[i] * i
        sum_b += abs(diff)
    
    return sum_a + sum_b

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    points = []
    idx = 1
    for _ in range(n):
        x = int(input[idx])
        y = int(input[idx + 1])
        points.append((x, y))
        idx += 2
    
    even = []
    odd = []
    for x, y in points:
        s = x + y
        if s % 2 == 0:
            even.append((x, y))
        else:
            odd.append((x, y))
    
    total = compute_sum(even) + compute_sum(odd)
    print(total)

if __name__ == '__main__':
    main()