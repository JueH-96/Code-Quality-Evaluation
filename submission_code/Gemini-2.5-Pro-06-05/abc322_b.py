# YOUR CODE HERE
import sys

def main():
  """
  Reads two strings S and T and determines if S is a prefix, a suffix, both, or neither of T.
  """
  # Read the lengths N and M from the first line of standard input.
  # Although not strictly necessary for the logic below, we consume them
  # as per the problem's input format.
  try:
    n, m = map(int, sys.stdin.readline().split())
  except (ValueError, IndexError):
    # Handles potential empty lines or incorrect formatting, though not
    # expected under problem constraints.
    return
  
  # Read the strings S and T from the following two lines.
  # .strip() removes any trailing newline characters.
  s = sys.stdin.readline().strip()
  t = sys.stdin.readline().strip()
  
  # Check if S is a prefix of T using the built-in startswith() method.
  is_prefix = t.startswith(s)
  
  # Check if S is a suffix of T using the built-in endswith() method.
  is_suffix = t.endswith(s)
  
  # Use a conditional structure to determine and print the output
  # based on the boolean values of is_prefix and is_suffix.
  if is_prefix and is_suffix:
    # S is both a prefix and a suffix.
    print(0)
  elif is_prefix:
    # S is a prefix, but not a suffix (due to the preceding 'if' being false).
    print(1)
  elif is_suffix:
    # S is a suffix, but not a prefix (due to the preceding 'if'/'elif' being false).
    print(2)
  else:
    # S is neither a prefix nor a suffix.
    print(3)

if __name__ == "__main__":
  main()