import itertools
import math

def is_palindrome_num(num):
    s = str(num)
    return s == s[::-1]

def no_zero_num(num):
    s = str(num)
    return '0' not in str(num)

def get_divisors(n):
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return list(divs)

# Precompute v_to_x and v_set
v_to_x = {}
for d in range(1, 7):  # 1 to 6 digits
    for comb in itertools.product(range(1, 10), repeat=d):  # digits 1-9
        num_str = ''.join(map(str, comb))
        x = int(num_str)
        rev_num_str = num_str[::-1]
        y = int(rev_num_str)
        v = x * y
        if v not in v_to_x:  # store one x for each v
            v_to_x[v] = x
v_set = set(v_to_x.keys())

# Read N from stdin
N = int(input())

found = False

# One-factor check
if no_zero_num(N) and is_palindrome_num(N):
    print(str(N))
    found = True
else:
    # Two-factor check
    if N in v_to_x:
        x = v_to_x[N]
        rev_x = int(str(x)[::-1])
        print(f"{x}*{rev_x}")
        found = True
    else:
        # Three-factor check
        for v, x in v_to_x.items():
            if N % v == 0:
                b = N // v
                if no_zero_num(b) and is_palindrome_num(b):
                    y = int(str(x)[::-1])
                    print(f"{x}*{b}*{y}")
                    found = True
                    break  # Break after finding a solution
        if not found:
            # Five-factor check
            div_n = get_divisors(N)
            for s in div_n:
                if found:
                    break  # Skip if solution already found
                for i in range(1, int(math.sqrt(s)) + 1):
                    if found:
                        break  # Skip if solution already found
                    if s % i == 0:
                        v1 = i
                        v2 = s // i
                        if v1 in v_set and v2 in v_set:
                            x_p = v_to_x[v1]
                            x_q = v_to_x[v2]
                            r = N // s
                            if no_zero_num(r) and is_palindrome_num(r):
                                p_str = str(x_p)
                                e_str = p_str[::-1]
                                q_str = str(x_q)
                                d_str = q_str[::-1]
                                r_str = str(r)
                                s_out = p_str + "*" + q_str + "*" + r_str + "*" + d_str + "*" + e_str
                                print(s_out)
                                found = True
                                break  # Break inner loop

if not found:
    print("-1")