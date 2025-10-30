def solve_case():
    S = input().strip()
    X = input().strip()
    Y = input().strip()
    
    # Count S and T in both patterns
    count_S_X = X.count('0')
    count_T_X = X.count('1')
    count_S_Y = Y.count('0')
    count_T_Y = Y.count('1')
    
    # Length constraint: count_S_X * |S| + count_T_X * |T| = count_S_Y * |S| + count_T_Y * |T|
    # Rearranging: (count_S_X - count_S_Y) * |S| = (count_T_Y - count_T_X) * |T|
    
    if count_T_Y == count_T_X:
        if count_S_X != count_S_Y:
            return "No"
        # T can have any length, try a reasonable range
        for T_len in range(min(len(S) + 1, 20)):
            if check_T_length(S, X, Y, T_len):
                return "Yes"
        return "No"
    else:
        # T has a specific required length
        if (count_S_X - count_S_Y) * len(S) % (count_T_Y - count_T_X) != 0:
            return "No"
        T_len = (count_S_X - count_S_Y) * len(S) // (count_T_Y - count_T_X)
        if T_len < 0:
            return "No"
        return "Yes" if check_T_length(S, X, Y, T_len) else "No"

def check_T_length(S, X, Y, T_len):
    # Create templates where S is substituted and T is represented by placeholders
    template_X = []
    template_Y = []
    
    for char in X:
        if char == '0':
            template_X.extend(S)
        else:
            template_X.extend(['T'] * T_len)
    
    for char in Y:
        if char == '0':
            template_Y.extend(S)
        else:
            template_Y.extend(['T'] * T_len)
    
    if len(template_X) != len(template_Y):
        return False
    
    # Determine constraints on T
    T_constraints = {}
    
    for i in range(len(template_X)):
        char_X = template_X[i]
        char_Y = template_Y[i]
        
        if char_X != 'T' and char_Y != 'T':
            # Both are concrete characters
            if char_X != char_Y:
                return False
        elif char_X == 'T' and char_Y != 'T':
            # X has T, Y has concrete character
            T_index = count_T_before(template_X, i) % T_len
            if T_index in T_constraints and T_constraints[T_index] != char_Y:
                return False
            T_constraints[T_index] = char_Y
        elif char_X != 'T' and char_Y == 'T':
            # X has concrete character, Y has T
            T_index = count_T_before(template_Y, i) % T_len
            if T_index in T_constraints and T_constraints[T_index] != char_X:
                return False
            T_constraints[T_index] = char_X
        # If both are T, they automatically match
    
    return True

def count_T_before(template, pos):
    # Count how many T's appear before position pos
    count = 0
    for i in range(pos):
        if template[i] == 'T':
            count += 1
    return count

def main():
    t = int(input())
    for _ in range(t):
        print(solve_case())

if __name__ == "__main__":
    main()