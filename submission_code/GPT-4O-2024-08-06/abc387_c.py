# YOUR CODE HERE
def is_snake_number(n):
    s = str(n)
    top_digit = s[0]
    for digit in s[1:]:
        if digit >= top_digit:
            return False
    return True

def count_snake_numbers(L, R):
    count = 0
    for number in range(L, R + 1):
        if is_snake_number(number):
            count += 1
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip()
    L, R = map(int, data.split())
    result = count_snake_numbers(L, R)
    print(result)