import sys
data = sys.stdin.read().split()
S = data[0]
N = int(data[1])
L = len(S)

def find_max(i, current_val, equal_so_far):
    if i == L:
        return current_val
    if not equal_so_far:
        max_suffix_val = 0
        weight = 1 << (L - 1 - i)
        for j in range(i, L):
            if S[j] == '?' or S[j] == '1':
                b = 1
            else:  # '0'
                b = 0
            max_suffix_val += b * weight
            weight >>= 1
        return current_val + max_suffix_val
    else:  # equal_so_far True
        shift_amount = L - 1 - i
        N_bit = (N >> shift_amount) & 1
        if S[i] == '?':
            if N_bit == 1:
                # try b=1 first
                new_val = current_val + (1 << shift_amount)
                new_equal_so_far = True  # b=1 == N_bit=1
                result = find_max(i + 1, new_val, new_equal_so_far)
                if result != -1:
                    return result
                # else try b=0
                new_val = current_val  # b=0
                new_equal_so_far = False  # b=0 != N_bit=1
                return find_max(i + 1, new_val, new_equal_so_far)
            else:  # N_bit == 0
                # only b=0
                new_val = current_val  # b=0
                new_equal_so_far = True  # b=0 == N_bit=0
                return find_max(i + 1, new_val, new_equal_so_far)
        elif S[i] == '0':
            b = 0
            # always can set
            new_val = current_val  # b=0
            new_equal_so_far = (0 == N_bit)
            return find_max(i + 1, new_val, new_equal_so_far)
        elif S[i] == '1':
            b = 1
            if b <= N_bit:  # N_bit == 1
                new_val = current_val + (1 << shift_amount)
                new_equal_so_far = True  # b==N_bit
                return find_max(i + 1, new_val, new_equal_so_far)
            else:
                return -1

# compute max_val
max_val_str = S.replace('?', '1')
max_val = int(max_val_str, 2)
if max_val <= N:
    print(max_val)
else:
    result = find_max(0, 0, True)
    if result == -1 or result > N:
        print(-1)
    else:
        print(result)