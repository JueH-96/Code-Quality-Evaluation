def solve():
    n = int(input())
    if n == 0:
        print(0)
        return
    
    index = n - 1
    if index == 0:
        print(0)
        return
        
    base5_digits = []
    if index == 0:
        base5_digits = [0]
    else:
        temp_index = index
        while temp_index > 0:
            base5_digits.append(temp_index % 5)
            temp_index //= 5
        base5_digits.reverse()
        
    if not base5_digits:
        base5_digits = [0]

    even_digits = []
    for digit in base5_digits:
        if digit == 0:
            even_digits.append('0')
        elif digit == 1:
            even_digits.append('2')
        elif digit == 2:
            even_digits.append('4')
        elif digit == 3:
            even_digits.append('6')
        elif digit == 4:
            even_digits.append('8')
            
    result_str = "".join(even_digits)
    print(result_str)

if __name__ == '__main__':
    solve()