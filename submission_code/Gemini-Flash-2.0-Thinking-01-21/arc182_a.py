def solve():
    n, q = map(int, input().split())
    operations = []
    for _ in range(q):
        operations.append(list(map(int, input().split())))
    
    memo = {}
    
    def get_valid_sequences_count(op_index, current_s_tuple):
        if op_index == q:
            return 1
        if (op_index, current_s_tuple) in memo:
            return memo[(op_index, current_s_tuple)]
        
        current_s = list(current_s_tuple)
        p_i, v_i = operations[op_index]
        
        count = 0
        
        # Choice 1
        condition1 = True
        if p_i > 0:
            if max(current_s[:p_i]) > v_i:
                condition1 = False
        if condition1:
            next_s_choice1 = list(current_s)
            for j in range(p_i):
                next_s_choice1[j] = v_i
            count = (count + get_valid_sequences_count(op_index + 1, tuple(next_s_choice1))) % 998244353
            
        # Choice 2
        condition2 = True
        if p_i <= n:
            if max(current_s[p_i-1:]) > v_i:
                condition2 = False
        if condition2:
            next_s_choice2 = list(current_s)
            for j in range(p_i - 1, n):
                next_s_choice2[j] = v_i
            count = (count + get_valid_sequences_count(op_index + 1, tuple(next_s_choice2))) % 998244353
            
        if not condition1 and not condition2:
            result = 0
        elif condition1 and not condition2:
            result = get_valid_sequences_count(op_index + 1, tuple(next_s_choice1)) if condition1 else 0
        elif not condition1 and condition2:
            result = get_valid_sequences_count(op_index + 1, tuple(next_s_choice2)) if condition2 else 0
        elif condition1 and condition2:
            result = (get_valid_sequences_count(op_index + 1, tuple(next_s_choice1)) + get_valid_sequences_count(op_index + 1, tuple(next_s_choice2))) % 998244353
        else:
            result = 0
            
        memo[(op_index, current_s_tuple)] = result
        return result
        
    initial_s = tuple([0] * n)
    result_count = get_valid_sequences_count(0, initial_s)
    print(result_count)

if __name__ == '__main__':
    solve()