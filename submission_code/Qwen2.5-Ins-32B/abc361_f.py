import math

def count_powerful_numbers(n):
    count = 0
    for base in range(2, int(n**0.5) + 1):
        power = 2
        while True:
            num = base ** power
            if num > n:
                break
            if num == 1:
                count += 1
                break
            count += 1
            power += 1
    return count + 1  # +1 for the number 1

if __name__ == "__main__":
    n = int(input().strip())
    print(count_powerful_numbers(n))