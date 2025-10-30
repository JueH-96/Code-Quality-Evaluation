# YOUR CODE HERE
import sys

def solve():
  """
  Reads the arrangement of plates and determines if rice is to the left of miso soup.
  """
  # Read the single line of input from stdin.
  # .strip() is used to remove any potential trailing newline characters.
  s = sys.stdin.readline().strip()

  # Find the index of the 'R' (Rice) plate.
  r_index = s.find('R')

  # Find the index of the 'M' (Miso soup) plate.
  m_index = s.find('M')

  # If the index of 'R' is smaller than the index of 'M', it means
  # 'R' appears earlier in the string (i.e., is to the left).
  if r_index < m_index:
    print("Yes")
  else:
    print("No")

solve()