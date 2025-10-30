def solve():
    n = int(input())
    if n == 0:
        print(0)
        return
    n -= 1
    base5_digits = []
    if n == 0:
        base5_digits = [0]
    else:
        while n > 0:
            digit = n % 5
            base5_digits.append(digit)
            n //= 5
        base5_digits.reverse()
    
    if not base5_digits:
        base5_digits = [0]
        
    mapping = {0: '0', 1: '2', 2: '4', 3: '6', 4: '8'}
    good_digits = [mapping[d] for d in base5_digits]
    result_str = "".join(good_digits)
    print(result_str)

if __name__ == '__main__':
    solve()