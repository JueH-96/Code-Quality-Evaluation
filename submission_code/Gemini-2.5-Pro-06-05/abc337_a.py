def main():
  """
  Reads match scores from standard input, calculates the total scores for
  Team Takahashi and Team Aoki, and prints the winner or "Draw".
  """
  
  try:
    # Read the number of matches from the first line of standard input.
    n = int(input())
  except (ValueError, IndexError):
    # This handles cases of empty or malformed input for N, though
    # the problem constraints guarantee valid input.
    return
  
  # Initialize the total scores for both teams to zero.
  takahashi_total_score = 0
  aoki_total_score = 0
  
  # Loop N times to read the scores for each match.
  for _ in range(n):
    # Read the two scores for the current match, separated by a space.
    # map(int, input().split()) reads the line, splits it into two strings,
    # and converts both strings to integers.
    x, y = map(int, input().split())
    
    # Add the score of the current match to each team's total.
    takahashi_total_score += x
    aoki_total_score += y
    
  # After iterating through all the matches, compare the final total scores.
  if takahashi_total_score > aoki_total_score:
    # If Team Takahashi's total score is greater, they win.
    print("Takahashi")
  elif aoki_total_score > takahashi_total_score:
    # If Team Aoki's total score is greater, they win.
    print("Aoki")
  else:
    # If the total scores are equal, it's a draw.
    print("Draw")

if __name__ == "__main__":
  main()