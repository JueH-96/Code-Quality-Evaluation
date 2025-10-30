import functools

def solve():
    n = int(input())
    a_values = []
    b_values = []
    for _ in range(n):
        a, b = map(int, input().split())
        a_values.append(a)
        b_values.append(b)
    
    indices = list(range(n))
    
    def compare_people(i_index, j_index):
        a_i = a_values[i_index]
        b_i = b_values[i_index]
        a_j = a_values[j_index]
        b_j = b_values[j_index]
        
        v_ij = a_i * b_j
        v_ji = a_j * b_i
        
        if v_ij > v_ji:
            return -1
        elif v_ij < v_ji:
            return 1
        else:
            if i_index < j_index:
                return -1
            elif i_index > j_index:
                return 1
            else:
                return 0
                
    sorted_indices = sorted(indices, key=functools.cmp_to_key(compare_people))
    
    result_person_numbers = [index + 1 for index in sorted_indices]
    print(*(result_person_numbers))

if __name__ == '__main__':
    solve()