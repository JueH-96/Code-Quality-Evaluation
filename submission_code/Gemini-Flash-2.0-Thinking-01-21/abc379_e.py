def solve():
    n = int(input())
    s = input()
    digits = []
    for char in s:
        digits.append(int(char))
    
    p_values = [0] * (n + 1)
    p_values[n] = 1
    for k in range(n - 1, 0, -1):
        p_values[k] = 10 * p_values[k + 1] + 1
        
    total_sum = 0
    for k in range(1, n + 1):
        digit = digits[k-1]
        contribution = digit * k * p_values[k]
        total_sum += contribution
        
    print(total_sum)

if __name__ == '__main__':
    solve()