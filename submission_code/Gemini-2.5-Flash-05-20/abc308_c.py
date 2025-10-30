import sys
from functools import cmp_to_key

# Custom comparison function for sorting people
# Each person is represented as a tuple: (A_i, B_i, person_number)
def compare_people(p1, p2):
    A1, B1, num1 = p1
    A2, B2, num2 = p2

    # Calculate total tosses for each person
    total1 = A1 + B1
    total2 = A2 + B2

    # Use cross-multiplication to compare success rates (A/Total)
    # This avoids floating-point precision issues with large numbers.
    # A1/total1 > A2/total2 is equivalent to A1*total2 > A2*total1
    val1_cross_product = A1 * total2
    val2_cross_product = A2 * total1

    # Primary sort criteria: Descending order of success rates
    if val1_cross_product > val2_cross_product:
        return -1  # p1 has a higher success rate, so it comes before p2
    elif val1_cross_product < val2_cross_product:
        return 1   # p2 has a higher success rate, so p1 comes after p2
    else:
        # Secondary sort criteria: Ascending order of person number (tie-breaker)
        if num1 < num2:
            return -1  # Success rates are equal, p1 has a smaller number, so it comes before p2
        # The problem statement implies person numbers are unique, so num1 == num2 won't happen for distinct people.
        else: # num1 > num2
            return 1   # Success rates are equal, p2 has a smaller number, so p1 comes after p2

def solve():
    # Read the number of people
    N = int(sys.stdin.readline())
    
    people_data = []
    for i in range(N):
        # Read A_i and B_i for each person
        A, B = map(int, sys.stdin.readline().split())
        # Store A, B, and the 1-indexed person number
        people_data.append((A, B, i + 1))

    # Sort the list of people_data using the custom comparison function
    # functools.cmp_to_key adapts a traditional comparison function (like compare_people)
    # to be used as a key for sorted() or list.sort().
    sorted_people = sorted(people_data, key=cmp_to_key(compare_people))

    # Extract only the person numbers from the sorted list
    result_numbers = [person_info[2] for person_info in sorted_people]

    # Print the result: space-separated person numbers
    print(*result_numbers)

# Ensure the solve function is called when the script is executed
if __name__ == '__main__':
    solve()