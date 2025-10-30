import collections

def solve():
  """
  Reads a string from stdin, determines if it is a "good string",
  and prints "Yes" or "No" to stdout.
  """
  # Read the input string from standard input.
  try:
    S = input()
  except EOFError:
    return

  # Step 1: Count the frequency of each character in the string S.
  # For S = "commencement", this results in:
  # Counter({'e': 3, 'm': 3, 'c': 2, 'n': 2, 'o': 1, 't': 1})
  char_counts = collections.Counter(S)

  # Step 2: Get the list of frequencies themselves. We are interested in how many
  # times each frequency value appears.
  # For S = "commencement", char_counts.values() gives [3, 3, 2, 2, 1, 1]
  # (the order is not guaranteed but does not matter).
  # We then count the occurrences of these numbers.
  # For [3, 3, 2, 2, 1, 1], this results in:
  # Counter({1: 2, 2: 2, 3: 2})
  # This tells us:
  # - 2 distinct letters appear exactly 1 time.
  # - 2 distinct letters appear exactly 2 times.
  # - 2 distinct letters appear exactly 3 times.
  # For any other number of occurrences `i` (e.g., i=4), 0 letters appear,
  # which is implicitly handled as they won't be keys in the counter.
  count_of_frequencies = collections.Counter(char_counts.values())

  # Step 3: Check if the string is "good".
  # A string is good if, for any number of occurrences `i`, there are
  # either 0 or 2 types of characters that appear `i` times.
  # This means all the values in our `count_of_frequencies` must be 2.
  is_good = all(count == 2 for count in count_of_frequencies.values())

  # Step 4: Print the result.
  if is_good:
    print("Yes")
  else:
    print("No")

solve()