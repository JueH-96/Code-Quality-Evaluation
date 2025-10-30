A, M, L, R = map(int, input().split())

def count_trees(A, M, L, R):
  """
  Counts the number of Christmas trees between L and R (inclusive)
  that are set up at intervals of M, starting from A.
  """

  # Find the first tree coordinate >= L
  first_tree = (L - A + M - 1) // M
  first_tree_coord = A + first_tree * M

  # Find the last tree coordinate <= R
  last_tree = (R - A) // M
  last_tree_coord = A + last_tree * M

  if first_tree_coord > R or last_tree_coord < L:
    return 0

  return last_tree - first_tree + 1

print(count_trees(A, M, L, R))