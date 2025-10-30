# YOUR CODE HERE
def find_largest_power_of_two(n):
    return n & -n

def divide_into_good_sequences(L, R):
    result = []
    while L < R:
        length = R - L
        power_of_two = find_largest_power_of_two(length)
        if power_of_two == 1:
            power_of_two = find_largest_power_of_two(L)
        next_R = L + power_of_two
        result.append((L, next_R))
        L = next_R
    return result

L, R = map(int, input().split())
sequences = divide_into_good_sequences(L, R)
print(len(sequences))
for l, r in sequences:
    print(l, r)