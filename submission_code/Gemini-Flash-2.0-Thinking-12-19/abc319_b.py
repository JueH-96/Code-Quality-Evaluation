def solve():
    n = int(input())
    divisors_in_range = []
    for j in range(1, 10):
        if n % j == 0:
            divisors_in_range.append(j)
    
    result_string = ""
    for i in range(n + 1):
        smallest_divisor = None
        for j in divisors_in_range:
            if n // j == 0:
                divisor_val = 0
            else:
                divisor_val = n // j
            if divisor_val == 0:
                if i % 1 == 0:
                    if smallest_divisor is None or j < smallest_divisor:
                        smallest_divisor = j
            elif i % divisor_val == 0:
                if smallest_divisor is None or j < smallest_divisor:
                    smallest_divisor = j
                    
        if smallest_divisor is not None:
            result_string += str(smallest_divisor)
        else:
            result_string += "-"
            
    print(result_string)

if __name__ == '__main__':
    solve()