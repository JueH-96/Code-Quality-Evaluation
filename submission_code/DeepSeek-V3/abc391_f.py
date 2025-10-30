import bisect

def solve():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, K = map(int, input[ptr:ptr+2])
    ptr +=2
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    C = list(map(int, input[ptr:ptr+N]))
    
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)
    C_sorted = sorted(C, reverse=True)
    
    left = 0
    right = 3 * 10**18
    
    def count_ge_x(x):
        total = 0
        for b in B_sorted:
            # The equation is (a_i + b) * (c_k + b) >= x + b*b
            target = x + b * b
            if target <=0:
                total += N * N
                continue
            # a_i + b must be positive (since A_i, B_j, C_k are >=1)
            # So (a_i + b) * (c_k + b) >= target
            # a_i + b >0, c_k +b >0
            # So for each a_i, find c_k >= (target)/(a_i +b) -b? No.
            # Let s_a = a_i + b, s_c = c_k + b. Then s_a * s_c >= target.
            # So s_c >= target / s_a, with s_a >0.
            # So for each s_a in the list (A_i + b), compute required s_c.
            # So for each a_i in A_sorted, compute s_a = a_i + b.
            # Then the required s_c is >= ceil(target / s_a)
            # So for the list of C_sorted + b, how many are >= val.
            # Precompute the list of C_sorted + b, sorted in descending order.
            # But since C is sorted in descending order, C_sorted + b is also descending.
            # So for s_a in s_a_list (sorted descending), for each s_a, the minimal s_c is (target + s_a -1) // s_a (since s_a * s_c >= target)
            s_a_list = [a + b for a in A_sorted]
            # s_a_list is in descending order (since A_sorted is descending)
            s_c_list = [c + b for c in C_sorted]
            # s_c_list is in descending order.
            cnt =0
            for s_a in s_a_list:
                if s_a ==0:
                    # but s_a = a_i + b >= 1 + 1 (since b >=1, a_i >=1)
                    pass
                if s_a <=0:
                    # impossible since a_i >=1, b >=1
                    pass
                min_s_c = (target + s_a -1) // s_a  # ceiling division
                # find number of elements in s_c_list >= min_s_c
                # since s_c_list is sorted in descending order
                # use bisect_left to find the first element < min_s_c, then the count is the index
                low =0
                high = len(s_c_list)
                # binary search for the first index where s_c_list[index] < min_s_c
                while low < high:
                    mid = (low + high) //2
                    if s_c_list[mid] >= min_s_c:
                        low = mid +1
                    else:
                        high = mid
                cnt += low
            total += cnt
            if total >= K:
                return total
        return total
    
    answer = 0
    while left <= right:
        mid = (left + right) //2
        cnt = count_ge_x(mid)
        if cnt >= K:
            answer = mid
            left = mid +1
        else:
            right = mid -1
    print(answer)

solve()