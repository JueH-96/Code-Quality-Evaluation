import sys

def find_nth_good_integer(n):
    count = 0
    num = 0
    while count < n:
        num += 1
        if all(int(digit) % 2 == 0 for digit in str(num)):
            count += 1
    return num

def main():
    n = int(sys.stdin.readline().strip())
    print(find_nth_good_integer(n))

if __name__ == "__main__":
    main()