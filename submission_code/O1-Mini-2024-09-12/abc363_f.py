# YOUR CODE HERE
def reverse_num(x):
    return int(str(x)[::-1])

def is_palindrome_str(s):
    return s == s[::-1]

def solve():
    import sys
    N = int(sys.stdin.read())
    str_N = str(N)
    if '0' not in str_N and is_palindrome_str(str_N):
        print(str_N)
        sys.exit()
    
    # Check m=2
    import math
    limit_m2 = int(math.isqrt(N)) +1
    for f1 in range(1, limit_m2):
        if N % f1 !=0:
            continue
        f2 = N // f1
        f1_str = str(f1)
        f2_str = str(f2)
        if '0' in f1_str or '0' in f2_str:
            continue
        if f2_str == f1_str[::-1]:
            print(f1_str + '*' + f2_str)
            sys.exit()
    
    # Check m=3
    for f1 in range(1, limit_m2):
        if N % f1 !=0:
            continue
        reverse_f1 = reverse_num(f1)
        if f1 * reverse_f1 ==0:
            continue
        if N % (f1 * reverse_f1) !=0:
            continue
        f2 = N // (f1 * reverse_f1)
        f2_str = str(f2)
        if '0' in f2_str:
            continue
        if is_palindrome_str(f2_str):
            f1_str = str(f1)
            reverse_f1_str = str(reverse_f1)
            print(f1_str + '*' + f2_str + '*' + reverse_f1_str)
            sys.exit()
    
    # Check m=4
    limit_m4 = int(math.isqrt(int(N**0.5))) +1
    for f1 in range(1, limit_m4):
        if N % f1 !=0:
            continue
        reverse_f1 = reverse_num(f1)
        if f1 * reverse_f1 ==0:
            continue
        if N % (f1 * reverse_f1) !=0:
            continue
        N1 = N // (f1 * reverse_f1)
        limit_f2 = int(math.isqrt(N1)) +1
        for f2 in range(1, limit_f2):
            if N1 % f2 !=0:
                continue
            f3 = N1 // f2
            reverse_f2 = reverse_num(f2)
            f2_str = str(f2)
            reverse_f2_str = str(reverse_f2)
            f3_str = str(f3)
            if '0' in f2_str or '0' in reverse_f2_str:
                continue
            if f3 == reverse_f2:
                print(f"{f1}*{f2}*{reverse_f2}*{reverse_f1}")
                sys.exit()
    
    # Check m=5
    limit_m5_f1 = int(N**0.25) +1
    for f1 in range(1, limit_m5_f1):
        if N % f1 !=0:
            continue
        reverse_f1 = reverse_num(f1)
        f1_str = str(f1)
        reverse_f1_str = str(reverse_f1)
        if '0' in f1_str or '0' in reverse_f1_str:
            continue
        if f1 * reverse_f1 ==0:
            continue
        N1 = N // (f1 * reverse_f1)
        limit_f2 = int(math.isqrt(N1)) +1
        for f2 in range(1, limit_f2):
            if N1 % f2 !=0:
                continue
            reverse_f2 = reverse_num(f2)
            f2_str = str(f2)
            reverse_f2_str = str(reverse_f2)
            if '0' in f2_str or '0' in reverse_f2_str:
                continue
            if f2 * reverse_f2 ==0:
                continue
            if N1 % (f2 * reverse_f2) !=0:
                continue
            f3 = N1 // (f2 * reverse_f2)
            f3_str = str(f3)
            if '0' in f3_str:
                continue
            if is_palindrome_str(f3_str):
                print(f"{f1}*{f2}*{f3}*{reverse_f2}*{reverse_f1}")
                sys.exit()
    
    # If no solution found
    print(-1)