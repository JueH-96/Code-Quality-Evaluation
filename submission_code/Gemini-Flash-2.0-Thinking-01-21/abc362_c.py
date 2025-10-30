def solve():
    n = int(input())
    ranges = []
    for _ in range(n):
        l, r = map(int, input().split())
        ranges.append({'l': l, 'r': r})
    
    min_sum = sum(r['l'] for r in ranges)
    max_sum = sum(r['r'] for r in ranges)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
    else:
        print("Yes")
        x = [r['l'] for r in ranges]
        current_sum = min_sum
        if current_sum < 0:
            target_increase = -current_sum
            for i in range(n):
                increase_amount = min(ranges[i]['r'] - x[i], target_increase)
                x[i] += increase_amount
                current_sum += increase_amount
                target_increase -= increase_amount
        elif current_sum > 0:
            target_decrease = current_sum
            for i in range(n):
                decrease_amount = min(x[i] - ranges[i]['l'], target_decrease)
                x[i] -= decrease_amount
                current_sum -= decrease_amount
                target_decrease -= decrease_amount
                
        print(*(x))

if __name__ == '__main__':
    solve()