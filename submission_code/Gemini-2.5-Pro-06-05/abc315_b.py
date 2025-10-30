import sys

def main():
  """
  Reads calendar specifications, finds the middle day of the year,
  and prints the result.
  """
  
  # Read the number of months from the first line of stdin.
  try:
    M = int(sys.stdin.readline())
    # Read the list of days for each month from the second line.
    D = list(map(int, sys.stdin.readline().split()))
  except (IOError, ValueError):
    # Handle potential empty input or invalid format, though problem constraints
    # suggest valid input.
    return

  # Calculate the total number of days in the year.
  total_days = sum(D)

  # Find the index of the middle day. Since total_days is guaranteed to be odd,
  # (total_days + 1) is even, so the division is exact.
  target_day_index = (total_days + 1) // 2

  # Iterate through the months to find which one contains the target day.
  # enumerate provides both the 0-based index `i` and the value `days_in_month`.
  for i, days_in_month in enumerate(D):
    # Check if the target day falls within the current month.
    if target_day_index <= days_in_month:
      # If it does, the month is `i + 1` (to convert from 0-based to 1-based)
      # and the day is the remaining `target_day_index`.
      month = i + 1
      day = target_day_index
      
      # Print the result in the required format "a b" and exit.
      print(month, day)
      return
    else:
      # If the target day is not in this month, subtract this month's days
      # from the target and continue to the next month.
      target_day_index -= days_in_month

if __name__ == "__main__":
  main()