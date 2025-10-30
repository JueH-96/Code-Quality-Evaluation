def solve():
  """
  Solves the problem of identifying a cake thief based on witness testimonies.
  """
  # Read the numbers of the two people who are known not to be the culprit.
  # The map(int, ...) function applies the int conversion to each item
  # from input().split(), which splits the input line by spaces.
  A, B = map(int, input().split())

  # The suspects are persons 1, 2, and 3.

  # If both witnesses name the same person, only one suspect is eliminated.
  # This leaves two possible culprits, so we cannot uniquely identify one.
  if A == B:
    print(-1)
  else:
    # If the witnesses name two different people, two suspects are eliminated.
    # This leaves exactly one suspect, who must be the culprit.
    # The sum of the suspects' numbers is 1 + 2 + 3 = 6.
    # By subtracting the numbers of the innocent people (A and B) from the sum,
    # we get the number of the remaining person, the culprit.
    culprit = 6 - A - B
    print(culprit)

# Execute the solution
solve()