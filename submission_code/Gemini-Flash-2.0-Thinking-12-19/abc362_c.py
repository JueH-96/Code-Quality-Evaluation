def solve():
    n = int(input())
    lr_pairs = []
    for _ in range(n):
        l, r = map(int, input().split())
        lr_pairs.append({'L': l, 'R': r})
    
    min_sum = sum([pair['L'] for pair in lr_pairs])
    max_sum = sum([pair['R'] for pair in lr_pairs])
    
    if min_sum > 0 or max_sum < 0:
        print("No")
    else:
        print("Yes")
        x_values = [pair['L'] for pair in lr_pairs]
        current_sum = sum(x_values)
        increase_needed = -current_sum
        for i in range(n):
            if increase_needed <= 0:
                break
            possible_increase = lr_pairs[i]['R'] - x_values[i]
            actual_increase = min(increase_needed, possible_increase)
            x_values[i] += actual_increase
            increase_needed -= actual_increase
            
        print(*(x_values))

if __name__ == '__main__':
    solve()