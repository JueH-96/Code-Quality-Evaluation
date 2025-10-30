def is_snake_number(n):
    str_n = str(n)
    top_digit = str_n[0]
    return all(top_digit > digit for digit in str_n[1:])

def count_snake_numbers(L, R):
    count = 0
    for number in range(L, R + 1):
        if is_snake_number(number):
            count += 1
    return count

import sys
input = sys.stdin.read

def main():
    data = input().strip()
    L, R = map(int, data.split())
    result = count_snake_numbers(L, R)
    print(result)

if __name__ == "__main__":
    main()