# YOUR CODE HERE

def check_conditions(numbers):
    for i in range(1, len(numbers)):
        if numbers[i] < numbers[i - 1] or numbers[i] < 100 or numbers[i] > 675 or numbers[i] % 25 != 0:
            return "No"
    return "Yes"

numbers = list(map(int, input().split()))
print(check_conditions(numbers))