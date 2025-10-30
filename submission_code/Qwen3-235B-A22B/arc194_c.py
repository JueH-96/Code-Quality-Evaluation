def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    elements_S = [i for i in range(N) if A[i] != B[i]]
    m = len(elements_S)
    
    if m == 0:
        print(0)
        return
    
    fixed_part_sum_not_S = 0
    for i in range(N):
        if A[i] == B[i]:
            fixed_part_sum_not_S += C[i] * A[i] * m
    
    sum_S = 0
    w_list = []
    for idx in elements_S:
        bi = B[idx]
        ai = A[idx]
        ci = C[idx]
        term = ci * (bi * (m + 1) - ai)
        sum_S += term
        w = ci * (ai - bi)
        w_list.append(w)
    
    fixed_part = fixed_part_sum_not_S + sum_S
    
    w_list.sort(reverse=True)
    
    sum_variable = 0
    for i in range(m):
        sum_variable += w_list[i] * (i + 1)
    
    total = fixed_part + sum_variable
    print(total)

if __name__ == "__main__":
    main()