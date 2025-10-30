import sys

def main():
  """
  Reads input, simulates the hand movements on the ring, and prints the minimum total operations.
  """
  # Read problem parameters N (size of the ring) and Q (number of instructions)
  try:
    N, Q = map(int, sys.stdin.readline().split())
  except (IOError, ValueError):
    # Handle cases with no input
    return

  # Initialize the positions of the left and right hands as per the problem statement
  l_pos = 1
  r_pos = 2

  # Initialize the accumulator for the total number of operations
  total_ops = 0

  # Process each of the Q instructions sequentially
  for _ in range(Q):
    try:
      # Read the instruction (Hand to move and Target position)
      H, T_str = sys.stdin.readline().split()
      T = int(T_str)
    except (IOError, ValueError):
      # Handle empty lines or malformed input at the end of a file
      break
      
    # Identify the current position of the moving hand and the stationary hand (the obstacle)
    if H == 'L':
      moving_pos = l_pos
      obstacle_pos = r_pos
    else:  # H == 'R'
      moving_pos = r_pos
      obstacle_pos = l_pos

    # If the target is the same as the current position, no operations are needed.
    if T == moving_pos:
      continue

    # Calculate the clockwise distance from the moving hand's current position to the target.
    # The formula `(b - a + N) % N` correctly gives the clockwise distance
    # on a circle of size N for 1-based indices a and b.
    dist_to_target_cw = (T - moving_pos + N) % N
    
    # Calculate the clockwise distance from the moving hand to the other (stationary) hand.
    dist_to_obstacle_cw = (obstacle_pos - moving_pos + N) % N

    # The clockwise path from `moving_pos` to `T` is unblocked if the obstacle
    # is not on that path. This is true if the clockwise distance to the target is
    # shorter than the clockwise distance to the obstacle.
    if dist_to_target_cw < dist_to_obstacle_cw:
      # The clockwise path is the shortest unblocked path.
      ops_this_turn = dist_to_target_cw
    else:
      # The clockwise path is blocked, so we must take the counter-clockwise path.
      # Its length is the total circumference (N) minus the clockwise path length.
      ops_this_turn = N - dist_to_target_cw
    
    # Add the number of operations for this instruction to the total.
    total_ops += ops_this_turn

    # Update the position of the hand that just moved.
    if H == 'L':
      l_pos = T
    else:
      r_pos = T

  # Print the final total number of operations.
  print(total_ops)

if __name__ == "__main__":
  main()