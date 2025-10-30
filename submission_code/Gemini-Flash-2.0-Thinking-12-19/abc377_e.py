def solve():
    n, k = map(int, input().split())
    p_initial = list(map(int, input().split()))
    current_p = tuple(p_initial)
    permutations_history = [current_p]
    
    while True:
        next_p_list = [0] * n
        current_p_tuple = permutations_history[-1]
        current_p_array = list(current_p_tuple)
        for i in range(n):
            next_p_list[i] = current_p_array[current_p_array[i]-1]
        next_p_tuple = tuple(next_p_list)
        if next_p_tuple in permutations_history:
            start_index = permutations_history.index(next_p_tuple)
            period = len(permutations_history) - start_index
            first_occurrence_index = start_index
            if k <= first_occurrence_index:
                result_permutation = permutations_history[k]
            else:
                effective_index = (k - first_occurrence_index) % period
                result_permutation = permutations_history[first_occurrence_index + effective_index]
            print(*(list(result_permutation)))
            return
        else:
            permutations_history.append(next_p_tuple)

if __name__ == '__main__':
    solve()