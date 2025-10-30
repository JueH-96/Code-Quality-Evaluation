def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    value_indices = {}
    for i in range(n):
        value = a[i]
        if value not in value_indices:
            value_indices[value] = []
        value_indices[value].append(i + 1)
        
    total_sum = 0
    for value in value_indices:
        indices = value_indices[value]
        previous_index = 0
        for current_index in indices:
            contribution = (current_index - previous_index) * (n - current_index + 1)
            total_sum += contribution
            previous_index = current_index
            
    print(total_sum)

if __name__ == '__main__':
    solve()