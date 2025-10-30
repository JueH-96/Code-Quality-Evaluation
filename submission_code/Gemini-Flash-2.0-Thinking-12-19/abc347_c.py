def solve():
    n, a, b = map(int, input().split())
    d = list(map(int, input().split()))
    week_length = a + b
    possible_start_indices = set()
    for r in range(a):
        possible_start_index = (r - d[0]) % week_length
        possible_start_indices.add(possible_start_index)
    
    for i in range(1, n):
        next_possible_indices = set()
        for r in range(a):
            day_index = (r - d[i]) % week_length
            if day_index in possible_start_indices:
                next_possible_indices.add(day_index)
        possible_start_indices = next_possible_indices
        
    if len(possible_start_indices) > 0:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()