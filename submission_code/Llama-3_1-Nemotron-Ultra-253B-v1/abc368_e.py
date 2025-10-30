import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, M, X1 = map(int, input[ptr:ptr+3])
    ptr += 3
    trains = []
    for i in range(M):
        A = int(input[ptr])
        B = int(input[ptr+1])
        S = int(input[ptr+2])
        T = int(input[ptr+3])
        ptr += 4
        trains.append((T, A, B, S, i + 1))
    
    trains.sort()
    city_info = {}
    X = [0] * (M + 1)
    X[1] = X1

    for T_j, A_j, B_j, S_j, j in trains:
        if j == 1:
            x = X1
        else:
            x = 0
            if A_j in city_info:
                lst = city_info[A_j]
                idx = bisect.bisect_right(lst, (S_j, float('inf'))) - 1
                if idx >= 0:
                    x = max(x, lst[idx][1] - S_j)
        X[j] = max(x, 0)
        
        current_list = city_info.get(B_j, [])
        bisect.insort(current_list, (T_j, T_j + X[j]))
        city_info[B_j] = current_list
    
    print(' '.join(map(str, X[2:M+1])))

main()