# YOUR CODE HERE

def is_good_string(s):
    letter_counts = {}
    for letter in s:
        if letter in letter_counts:
            letter_counts[letter] += 1
        else:
            letter_counts[letter] = 1
    count_of_one_letters = sum(1 for count in letter_counts.values() if count == 1)
    count_of_two_letters = sum(1 for count in letter_counts.values() if count == 2)
    return count_of_one_letters == 0 and count_of_two_letters == 0

s = input()
print("Yes" if is_good_string(s) else "No")