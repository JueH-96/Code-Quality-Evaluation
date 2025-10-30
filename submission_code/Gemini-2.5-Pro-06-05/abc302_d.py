import sys
import bisect

def run():
  """
  Reads input, solves the problem, and prints the output.
  """
  # Use sys.stdin.readline for faster I/O, which is important for large inputs.
  input = sys.stdin.readline
  
  try:
    # Read problem parameters N, M, D
    line = input()
    # Handle empty lines which might be present in some environments
    if not line.strip():
        return
    N, M, D = map(int, line.split())

    # Read the list of gift values for Aoki
    A = list(map(int, input().split()))
    
    # Read the list of gift values for Snuke
    B = list(map(int, input().split()))
  except (ValueError, IndexError):
    # Exit gracefully if input is malformed or empty
    return

  # Sort list B to enable efficient binary searching
  B.sort()

  max_sum = -1

  # Iterate through each gift 'a' from Aoki's list
  for a in A:
    # For a fixed 'a', we need a gift 'b' such that a - D <= b <= a + D.
    # To maximize the sum 'a + b', we need to find the largest 'b' in this range.
    
    # The largest possible 'b' must satisfy b <= a + D.
    # We use binary search to find the largest element in B that is at most a + D.
    # `bisect_right` returns an index `idx` such that for any j < idx, B[j] <= (a + D).
    idx = bisect.bisect_right(B, a + D)

    # If idx is 0, it means no element in B is less than or equal to a + D.
    # Therefore, no valid partner 'b' exists for this 'a'.
    if idx == 0:
      continue

    # The best candidate for 'b' is the element at the preceding index.
    b_candidate = B[idx - 1]

    # Now we check if this candidate also satisfies the lower bound: b >= a - D.
    # The original condition is abs(a - b) <= D. We have already ensured b <= a + D
    # (implicitly by how we found it), so we only need to check b >= a - D.
    if b_candidate >= a - D:
      # If the condition holds, we have a valid pair.
      current_sum = a + b_candidate
      # Update the maximum sum if the current sum is greater.
      max_sum = max(max_sum, current_sum)

  # Print the final maximum sum. If no valid pair was found, it remains -1.
  print(max_sum)

run()