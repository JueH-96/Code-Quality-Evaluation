import collections

# Read the input string S
S = input()

# 1. Count frequencies of each character in S
# Example for S="commencement":
# char_counts = {'c': 2, 'o': 1, 'm': 3, 'e': 3, 'n': 2, 't': 1}
char_counts = collections.Counter(S)

# 2. Get the list of these frequencies (how many times each character appeared)
# Example for S="commencement":
# character_occurrence_counts_list = [2, 1, 3, 3, 2, 1] (actual order might vary)
character_occurrence_counts_list = char_counts.values()

# 3. Count how many times each frequency value itself appears.
# This tells us, for each count i, how many distinct characters (N(i)) appear i times.
# Example for S="commencement":
# N(1) = 2 (for 'o', 't')
# N(2) = 2 (for 'c', 'n')
# N(3) = 2 (for 'e', 'm')
# This step results in: counts_of_frequencies = {1: 2, 2: 2, 3: 2}
counts_of_frequencies = collections.Counter(character_occurrence_counts_list)

# 4. Check the condition for a good string.
# For any integer i >= 1, N(i) must be 0 or 2.
# The `counts_of_frequencies` dictionary stores N(i) for all i where N(i) > 0.
# So, every value in `counts_of_frequencies.values()` must be 2.
# (If N(i) = 0 for some i, then i is not a key in counts_of_frequencies, which is fine.)
is_good_string = True
for num_chars_sharing_a_frequency in counts_of_frequencies.values():
    # num_chars_sharing_a_frequency is N(i) for some i where N(i) > 0.
    # This N(i) must be 2.
    if num_chars_sharing_a_frequency != 2:
        is_good_string = False
        break

if is_good_string:
    print("Yes")
else:
    print("No")