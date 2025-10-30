def solve():
    n = int(input())
    a = list(map(int, input().split()))
    occurrence_indices = {}
    for index, value in enumerate(a):
        if value not in occurrence_indices:
            occurrence_indices[value] = []
        occurrence_indices[value].append(index + 1)
    
    f_values = {}
    for i in range(1, n + 1):
        indices_list = occurrence_indices[i]
        f_values[i] = indices_list[1]
        
    numbers_with_f_values = []
    for i in range(1, n + 1):
        numbers_with_f_values.append((f_values[i], i))
        
    numbers_with_f_values.sort(key=lambda pair: pair[0])
    
    sorted_numbers = [number for f_value, number in numbers_with_f_values]
    
    print(*(sorted_numbers))

if __name__ == '__main__':
    solve()