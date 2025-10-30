# YOUR CODE HERE
import collections

# Read the input string
S = input()

# Count the frequency of each character
# char_counts maps char -> frequency (e.g., {'a': 2, 'b': 1})
char_counts = collections.Counter(S)

# Get the frequency of each character
# frequencies is a list of integers, each representing the count of a unique character
# e.g., from {'a': 2, 'b': 1, 'c': 1}, this list would be [2, 1, 1]
frequencies = list(char_counts.values())

# Count how many characters have each specific frequency
# freq_of_frequencies maps frequency_value -> number_of_characters_with_that_frequency
# e.g., from [2, 1, 1], this maps to {2: 1, 1: 2}, meaning 1 character appears 2 times, 2 characters appear 1 time
freq_of_frequencies = collections.Counter(frequencies)

# Check the condition: For every frequency i >= 1, the number of different letters
# that appear exactly i times must be 0 or 2.
# In freq_of_frequencies, the key is the frequency i (>= 1), and the value is
# the number of different letters that appear exactly i times.
# If a frequency i does not appear in freq_of_frequencies as a key, it means
# zero letters appear exactly i times, which satisfies the condition (exactly zero).
# So, we only need to check the frequencies that *do* appear as keys in freq_of_frequencies.
# For these frequencies, the number of letters must be exactly 2.
is_good = True
for freq_value, num_chars in freq_of_frequencies.items():
    # freq_value is the frequency (e.g., 1, 2, 3...)
    # num_chars is the count of characters that appear 'freq_value' times.
    # This 'num_chars' must be exactly 2 for the string to be good, because
    # any key in freq_of_frequencies implies num_chars >= 1.
    if num_chars != 2:
        is_good = False
        break # Found a frequency where the character count is not 2 (and not 0), so it's not good.

# Print the result
if is_good:
    print("Yes")
else:
    print("No")