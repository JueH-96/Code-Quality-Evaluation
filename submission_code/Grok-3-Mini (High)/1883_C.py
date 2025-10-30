import sys
data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1
for _ in range(t):
    n = int(data[index])
    index += 1
    k = int(data[index])
    index += 1
    array = list(map(int, data[index:index + n]))
    index += n
    if k == 4:
        num_even = 0
        min_cost_div4 = 4  # Initialize to a value larger than maximum possible cost (3)
        for num in array:
            rem = num % 4
            cost_div4_num = (4 - rem) % 4
            if cost_div4_num < min_cost_div4:
                min_cost_div4 = cost_div4_num
            if num % 2 == 0:
                num_even += 1
        if num_even >= 2:
            cost_two_even = 0
        elif num_even == 1:
            cost_two_even = 1
        else:
            cost_two_even = 2
        ans = min(min_cost_div4, cost_two_even)
    else:  # k=2,3,5
        min_cost = min((k - (num % k)) % k for num in array)
        ans = min_cost
    print(ans)