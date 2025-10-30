# YOUR CODE HERE
import sys

# Function to calculate total pills on a given day D
def calculate_pills(day, medicines):
    """
    Calculates the total number of pills Takahashi has to take on a specific day.

    Args:
        day (int): The day number (starting from 1).
        medicines (list of tuples): A list where each tuple is (a_i, b_i)
                                     representing the duration and pills per day
                                     for a medicine.

    Returns:
        int: The total number of pills on the given day.
    """
    total_pills = 0
    for a, b in medicines:
        # If the prescription for this medicine (lasting a_i days) includes 'day'
        if day <= a:
            total_pills += b
    return total_pills

def solve():
    """
    Finds the first day (on or after day 1) when the total number of pills
    Takahashi has to take is K or less.
    """
    # Read N (number of medicines) and K (maximum allowed pills per day)
    N, K = map(int, sys.stdin.readline().split())

    # Read the medicine details (a_i: duration, b_i: pills per day) for N medicines
    medicines = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        medicines.append((a, b))

    # The problem asks for the first day X >= 1 such that the total pills on day X is <= K.
    # Let P(X) be the total pills on day X. P(X) is a non-increasing function of X.
    # We are looking for the minimum X >= 1 such that P(X) <= K.
    # This structure allows us to use binary search on the day number.

    # We need a search range [low, high) for the binary search.
    # The minimum possible day is 1. So, low = 1.
    # We know that for any day D strictly greater than the maximum duration max(a_i),
    # the total number of pills P(D) is 0.
    # The maximum value of a_i is 10^9. So, day 10^9 + 1 will have 0 pills.
    # Since K >= 0, P(10^9 + 1) = 0 <= K is always true.
    # Therefore, the answer must be in the range [1, 10^9 + 1].
    # We set our binary search range to include all integers from 1 up to 10^9 + 1.
    # In a [low, high) binary search template, this means high = (10^9 + 1) + 1 = 10^9 + 2.
    low = 1
    high = 10**9 + 2 
    
    # Initialize 'ans' to a value outside the valid range [1, 10^9 + 1].
    # If the binary search finds a day satisfying the condition, 'ans' will be updated.
    # Since we search for the minimum day, 'ans' will correctly store the minimum.
    ans = high 

    # Perform binary search
    # The loop continues as long as the search range [low, high) is not empty.
    while low < high:
        # Calculate the middle day of the current search range.
        mid = low + (high - low) // 2 

        # Calculate the total number of pills on the 'mid' day.
        current_pills = calculate_pills(mid, medicines)

        # Check if the condition (total pills <= K) is met on 'mid' day.
        if current_pills <= K:
            # If the condition is met, 'mid' is a possible answer.
            # We record 'mid' as the current best candidate.
            ans = mid
            # Since we are looking for the *smallest* such day, the actual first day
            # might be 'mid' or an earlier day. We search in [low, mid).
            high = mid 
        else:
            # If the condition is NOT met (current_pills > K), then 'mid' has too many pills.
            # The first day with <= K pills must be strictly later than 'mid'.
            # We search in [mid + 1, high).
            low = mid + 1

    # After the loop finishes (low == high), 'ans' holds the minimum day X >= 1
    # that satisfied calculate_pills(X, medicines) <= K.
    print(ans)

# Execute the main solve function to run the program
solve()