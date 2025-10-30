import sys

def solve():
    S = sys.stdin.readline().strip()

    # The first character is the first digit
    num1_char = S[0]
    # The third character is the second digit
    num2_char = S[2]

    # Convert character digits to integers
    num1 = int(num1_char)
    num2 = int(num2_char)

    # Calculate the product
    product = num1 * num2

    # Print the result
    print(product)

if __name__ == '__main__':
    solve()