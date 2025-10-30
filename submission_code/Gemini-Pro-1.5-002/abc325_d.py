# YOUR CODE HERE
def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        t, d = map(int, input().split())
        intervals.append((t, t + d))

    intervals.sort(key=lambda x: x[1])

    count = 0
    last_print_time = 0
    for start, end in intervals:
        if start >= last_print_time + 1:
            count += 1
            last_print_time = start
        elif end >= last_print_time + 1:
            count +=1
            last_print_time = last_print_time + 1


    print(count)

solve()