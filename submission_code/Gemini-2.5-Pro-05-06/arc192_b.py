def solve():
    N = int(input())
    A = list(map(int, input().split()))

    if N == 1:
        print("Fennec")
        return

    has_A_i_eq_1 = False
    for x in A:
        if x == 1:
            has_A_i_eq_1 = True
            break
    
    if has_A_i_eq_1:
        if N % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")
    else: # All A_i >= 2
        sum_A = sum(A)
        if sum_A % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")

if __name__ == '__main__':
    solve()