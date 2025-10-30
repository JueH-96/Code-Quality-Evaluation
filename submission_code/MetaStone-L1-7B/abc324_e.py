import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = input[ptr]
    ptr += 1
    
    len_T = len(T)
    a_list = []
    b_list = []
    
    for _ in range(N):
        S = input[ptr]
        ptr += 1
        # Compute a
        a_counter = 0
        for c in S:
            if a_counter < len_T and c == T[a_counter]:
                a_counter += 1
        a = a_counter
        a_list.append(a)
        
        # Compute b
        T_rev = T[::-1]
        b_counter = 0
        for c in reversed(S):
            if b_counter < len_T and c == T_rev[b_counter]:
                b_counter += 1
        b = b_counter
        b_list.append(b)
    
    # Sort a_list
    a_list.sort()
    
    count = 0
    for bj in b_list:
        x = len_T - bj
        if x <= 0:
            count += N
        else:
            idx = bisect.bisect_left(a_list, x)
            count += len(a_list) - idx
    
    print(count)

if __name__ == '__main__':
    main()