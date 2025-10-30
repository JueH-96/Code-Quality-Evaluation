def solve():
    import sys

    N = sys.stdin.readline().strip()
    # Convert the input string to a list of digits
    digits = list(N)

    # We only need to check if each digit is strictly greater than the next
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i+1]:
            print("No")
            return

    print("Yes")

def main():
    solve()

if __name__ == "__main__":
    main()