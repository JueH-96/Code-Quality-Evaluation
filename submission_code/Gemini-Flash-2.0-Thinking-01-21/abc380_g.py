def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    
    if k == 1:
        inversion_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    inversion_count += 1
        print(inversion_count % 998244353)
        return
        
    expected_inversions_sum = 0
    mod = 998244353
    inv_2 = pow(2, mod - 2, mod)
    inv_4 = pow(4, mod - 2, mod)
    inv_k = pow(k, mod - 2, mod) if k > 0 else 0
    inv_nk1 = pow(n - k + 1, mod - 2, mod) if n - k + 1 > 0 else 0
    
    for j in range(n):
        for l in range(j + 1, n):
            prob_sum = 0
            for i_start_index in range(1, n - k + 2):
                i = i_start_index - 1
                pj = p[j]
                pl = p[l]
                prob = 0
                if i <= j <= i + k - 1 and i <= l <= i + k - 1:
                    prob = inv_2
                elif i <= j <= i + k - 1 and l > i + k - 1:
                    count_greater = 0
                    for index in range(i, i + k):
                        if p[index] > pl:
                            count_greater += 1
                    prob = (count_greater * inv_k) % mod
                elif j < i and i <= l <= i + k - 1:
                    count_less = 0
                    for index in range(i, i + k):
                        if p[index] < pj:
                            count_less += 1
                    prob = (count_less * inv_k) % mod
                else:
                    if pj > pl:
                        prob = 1
                    else:
                        prob = 0
                prob_sum = (prob_sum + prob) % mod
            expected_prob = (prob_sum * inv_nk1) % mod
            expected_inversions_sum = (expected_inversions_sum + expected_prob) % mod
            
    print(expected_inversions_sum)

if __name__ == '__main__':
    solve()