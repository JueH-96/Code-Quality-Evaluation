def solve():
    n = int(input())
    s = input()
    digits = [int(d) for d in s]
    total_sum = 0
    powers_of_10 = [1] * (n + 2)
    for i in range(1, n + 2):
        powers_of_10[i] = powers_of_10[i-1] * 10
    
    for k in range(1, n + 1):
        v_k = digits[k-1]
        term_k = (v_k * k * (powers_of_10[n - k + 1] - 1)) // 9
        total_sum += term_k
        
    print(total_sum)

if __name__ == '__main__':
    solve()