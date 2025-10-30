import sys

def is_326_like(number):
    hundreds = number // 100
    tens = (number % 100) // 10
    ones = number % 10
    return hundreds * tens == ones

def find_smallest_326_like(N):
    for num in range(N, 920):  # 919 is the maximum possible 326-like number
        if is_326_like(num):
            return num

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    result = find_smallest_326_like(N)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()