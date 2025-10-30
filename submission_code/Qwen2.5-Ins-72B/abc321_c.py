# YOUR CODE HERE
def generate_321_like_numbers():
    from itertools import combinations
    
    digits = '9876543210'
    numbers = []
    
    for length in range(1, 11):
        for combo in combinations(digits, length):
            if combo[-1] != '0' or length == 1:
                numbers.append(int(''.join(combo)))
    
    numbers.sort()
    return numbers

def main():
    K = int(input())
    numbers = generate_321_like_numbers()
    print(numbers[K-1])

if __name__ == "__main__":
    main()