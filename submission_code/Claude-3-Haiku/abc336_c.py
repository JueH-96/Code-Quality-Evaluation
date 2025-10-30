def nth_smallest_good_integer(n):
    good_integers = []
    i = 0
    while len(good_integers) < n:
        if all(int(digit) % 2 == 0 for digit in str(i)):
            good_integers.append(i)
        i += 2
    return good_integers[n-1]

# Read input from stdin
n = int(input())

# Print the answer to stdout
print(nth_smallest_good_integer(n))