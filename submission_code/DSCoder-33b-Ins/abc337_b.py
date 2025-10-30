# YOUR CODE HERE

def is_extended_abc(s):
    a_count = s.count('A')
    b_count = s.count('B')
    c_count = s.count('C')

    if a_count == 0 or b_count == 0 or c_count == 0:
        return False

    if a_count + b_count + c_count != len(s):
        return False

    return True

s = input()
print('Yes' if is_extended_abc(s) else 'No')