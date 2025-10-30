from itertools import combinations

def generate_321_like_numbers():
    numbers = []
    # Generate all possible combinations of digits in strictly decreasing order
    for r in range(1, 10):  # r is the number of digits
        for digits in combinations(range(9, -1, -1), r):
            if digits[-1] == 0 and len(digits) > 1:
                continue  # Skip numbers like 10, 20, etc., but allow single-digit 0
            num = 0
            for d in digits:
                num = num * 10 + d
            numbers.append(num)
    # Sort the numbers
    numbers.sort()
    return numbers

def main():
    K = int(input())
    numbers = generate_321_like_numbers()
    print(numbers[K-1])

if __name__ == "__main__":
    main()