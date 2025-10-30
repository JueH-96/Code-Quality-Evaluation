def solve():
  """
  Solves the problem of finding the minimum points needed for person 1 to be the strongest.
  """
  try:
    # Read the number of people, N.
    N = int(input())

    # Read the list of programming ability scores, P.
    P = list(map(int, input().split()))
  except (IOError, ValueError):
    # Handle potential empty input or invalid format, though not expected by problem constraints.
    return

  # Edge case: If there is only one person, they are already the strongest.
  if N == 1:
    print(0)
    return

  # Get the score of person 1.
  p1_score = P[0]

  # Get the scores of all other people (from person 2 to N).
  other_scores = P[1:]

  # Find the maximum score among the other people.
  max_other_score = max(other_scores)

  # Check if person 1 is already strictly stronger than everyone else.
  if p1_score > max_other_score:
    # If so, no additional points are needed.
    print(0)
  else:
    # Otherwise, calculate the points needed.
    # The target score for person 1 must be 1 point more than the max of others.
    target_score = max_other_score + 1
    
    # The points to add is the difference between the target and current score.
    points_needed = target_score - p1_score
    print(points_needed)

solve()