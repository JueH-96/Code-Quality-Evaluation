# YOUR CODE HERE
def solve():
  """
  Solves the oyster problem by deducing the poisonous type based on
  the sickness status of two people.
  """
  # Read the two status strings from standard input.
  s_1, s_2 = input().split()

  # Initialize a set of all possible oyster types that could be the cause.
  suspects = {1, 2, 3, 4}

  # --- Update possibilities based on Takahashi's status ---
  # Takahashi ate oysters {1, 2}.
  # If he is sick, the culprit is in {1, 2}. Otherwise, it is in {3, 4}.
  if s_1 == 'sick':
    suspects &= {1, 2}
  else:
    suspects &= {3, 4}

  # --- Update possibilities based on Aoki's status ---
  # Aoki ate oysters {1, 3}.
  # If he is sick, the culprit is in {1, 3}. Otherwise, it is in {2, 4}.
  if s_2 == 'sick':
    suspects &= {1, 3}
  else:
    suspects &= {2, 4}

  # After considering both pieces of evidence, the problem guarantees that
  # only one suspect remains.
  # The pop() method removes and returns an element from the set. Since
  # there is only one element, this gives us our answer.
  print(suspects.pop())

solve()