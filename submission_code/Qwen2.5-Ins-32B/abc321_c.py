import itertools

def find_kth_321_like_number(k):
    # Generate all possible 321-like numbers
    digits = '9876543210'
    for length in range(1, 11):
        for combination in itertools.combinations(digits, length):
            number = int(''.join(combination))
            k -= 1
            if k == 0:
                return number

if __name__ == "__main__":
    k = int(input().strip())
    print(find_kth_321_like_number(k))