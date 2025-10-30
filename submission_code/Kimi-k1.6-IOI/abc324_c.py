import sys

def is_one_edit_deletion(long_s, short_s):
    if len(long_s) - len(short_s) != 1:
        return False
    i = j = 0
    count = 0
    len_short = len(short_s)
    len_long = len(long_s)
    while i < len_short and j < len_long:
        if short_s[i] == long_s[j]:
            i += 1
            j += 1
        else:
            j += 1
            count += 1
            if count > 1:
                return False
    count += (len_long - j)
    return count == 1

def is_valid(s, t_prime):
    len_s = len(s)
    len_t = len(t_prime)
    if len_s == len_t:
        diff = 0
        for a, b in zip(s, t_prime):
            if a != b:
                diff += 1
                if diff > 1:
                    return False
        return True
    elif len_s == len_t - 1:
        return is_one_edit_deletion(t_prime, s)
    elif len_s == len_t + 1:
        return is_one_edit_deletion(s, t_prime)
    else:
        return False

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    t_prime = data[1]
    s_list = data[2:2 + n]
    valid_indices = []
    for idx in range(n):
        if is_valid(s_list[idx], t_prime):
            valid_indices.append(idx + 1)
    print(len(valid_indices))
    if valid_indices:
        print(' '.join(map(str, valid_indices)))

if __name__ == '__main__':
    main()