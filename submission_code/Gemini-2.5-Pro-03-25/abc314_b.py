# YOUR CODE HERE
import sys

def solve():
    # Read the number of people
    n = int(sys.stdin.readline())

    # Store data for each person. Using a list of dictionaries is one way.
    # Each dictionary will store the person's ID (1-based), their bet count, and their set of bets.
    people_data = []
    # Loop from 1 to N to match person IDs (1, 2, ..., N) as used in the problem statement
    for i in range(1, n + 1):
        # Read the count of bets (C_i) for person i
        c = int(sys.stdin.readline())
        # Read the bets (A_{i, 1}, ..., A_{i, C_i}) for person i.
        # Convert the space-separated string of numbers into a list of integers.
        # Store the bets as a set for efficient membership checking (checking if X is in the set).
        a_set = set(map(int, sys.stdin.readline().split()))
        # Append the collected data for this person to our list
        people_data.append({'id': i, 'count': c, 'bets': a_set})

    # Read the winning number X
    x = int(sys.stdin.readline())

    # Identify people who bet on the winning number X
    # We store winners as a list of dictionaries, keeping their ID and total bet count.
    winners = []
    for person in people_data:
        # Check if the winning number X is present in the person's set of bets
        if x in person['bets']:
            # If they bet on X, add their ID and total bet count ('count') to the winners list
            winners.append({'id': person['id'], 'count': person['count']})

    # Check if any winners were found
    if not winners:
        # If the 'winners' list is empty, it means no one bet on X.
        # Print 0 as required by the problem statement for the count K.
        print(0)
        # As per Sample Output 2, when K=0, only print 0. No second line is printed.
        return # Exit the function

    # If there are winners, find the minimum bet count among them.
    # Initialize min_bet_count to positive infinity to ensure any real count found will be smaller.
    min_bet_count = float('inf')
    for winner in winners:
        # Update min_bet_count if a winner with a smaller bet count is found
        if winner['count'] < min_bet_count:
            min_bet_count = winner['count']
    
    # An alternative, more concise way to find the minimum using a generator expression:
    # min_bet_count = min(winner['count'] for winner in winners)

    # Now, filter the list of winners to find only those whose bet count matches the minimum count found.
    # Store the IDs of these final candidates.
    final_candidates_ids = []
    for winner in winners:
        # If a winner's bet count equals the minimum count, they are a final candidate.
        if winner['count'] == min_bet_count:
            final_candidates_ids.append(winner['id'])
            
    # An alternative, more concise way using a list comprehension:
    # final_candidates_ids = [winner['id'] for winner in winners if winner['count'] == min_bet_count]

    # The problem requires the output IDs (B_1, ..., B_K) to be sorted in ascending order.
    final_candidates_ids.sort()

    # Print the total number of final candidates (K) on the first line.
    print(len(final_candidates_ids))

    # Print the sorted IDs of the final candidates, separated by spaces, on the second line.
    # The * operator unpacks the list elements, passing them as separate arguments to print(),
    # which then prints them separated by spaces by default.
    print(*final_candidates_ids)

# Execute the main logic of the program
solve()