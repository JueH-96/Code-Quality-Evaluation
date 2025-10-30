def invert_case(char):
    if 'a' <= char <= 'z':
        return char.upper()
    elif 'A' <= char <= 'Z':
        return char.lower()
    else:
        return char

def solve_query(initial_s, k, operations_remaining):
    if operations_remaining == 0:
        return initial_s[k-1]
    current_length = len(initial_s)
    length_prev_s = current_length * (2**(operations_remaining - 1))
    if k <= length_prev_s:
        return solve_query(initial_s, k, operations_remaining - 1)
    else:
        char = solve_query(initial_s, k - length_prev_s, operations_remaining - 1)
        return invert_case(char)

if __name__ == '__main__':
    s = input()
    q = int(input())
    queries = list(map(int, input().split()))
    results = []
    num_operations = 60 # Sufficient number of operations
    for k in queries:
        result_char = solve_query(s, k, num_operations)
        results.append(result_char)
    print(*(results))