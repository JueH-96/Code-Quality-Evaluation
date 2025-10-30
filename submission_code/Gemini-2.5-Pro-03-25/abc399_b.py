# YOUR CODE HERE
import sys

def main():
    """
    Reads the number of participants and their scores, calculates their ranks based on the specified procedure,
    and prints the rank for each participant.

    The ranking procedure assigns ranks based on scores:
    1. Initialize rank counter `r = 1`.
    2. Find the maximum score `x` among unranked participants.
    3. Let `k` be the number of unranked participants with score `x`.
    4. Assign rank `r` to these `k` participants.
    5. Increment `r` by `k`.
    6. Repeat until all participants are ranked.

    This procedure is equivalent to sorting participants by score in descending order
    and assigning ranks based on position, handling ties appropriately. Participants
    with the same score receive the same rank, and the rank assigned is `1 + (number of people with strictly higher scores)`.
    """
    # Read the number of participants, N.
    n = int(sys.stdin.readline())
    # Read the scores of the N participants into a list P.
    p = list(map(int, sys.stdin.readline().split()))

    # Create a list of tuples, where each tuple contains (score, original_index).
    # The original_index is 0-based and corresponds to person i+1 (where person labels are 1 to N).
    items = []
    for i in range(n):
        items.append((p[i], i)) # Store score and its original 0-based index

    # Sort the list of items based on the score in descending order.
    # `key=lambda x: x[0]` specifies sorting based on the first element of the tuple (the score).
    # `reverse=True` ensures descending order (highest score first).
    # Python's `sorted()` is stable, meaning elements with the same key (score)
    # maintain their relative order from the original `items` list. This doesn't
    # affect the final rank calculation but is a property of the sort.
    sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

    # Create a list to store the final ranks for each participant.
    # The index of this list corresponds to the original 0-based index (0 to N-1).
    # Initialize all ranks to 0 (or any placeholder value).
    ranks = [0] * n

    # Variable to store the rank value currently being assigned.
    # This value represents the rank for the current group of participants with the same score.
    # It is updated only when a new score group (with a lower score) is encountered.
    current_rank_value = 0

    # Iterate through the sorted list of participants to assign ranks.
    # 'i' represents the 0-based index in the `sorted_items` list.
    for i in range(n):
        # Get the score and the original 0-based index of the current participant.
        score, original_index = sorted_items[i]

        # Check if this is the first participant in the sorted list (index i=0).
        if i == 0:
            # The participant(s) with the highest score get rank 1.
            current_rank_value = 1
            # Store the rank in the `ranks` list at the position corresponding to the original index.
            ranks[original_index] = current_rank_value
        else:
            # If not the first participant, compare their score with the previous participant in the sorted list.
            # Get the score of the previous participant (at index i-1).
            prev_score, _ = sorted_items[i-1] # Only the score is needed for comparison.

            # Compare the current participant's score with the previous one.
            if score == prev_score:
                # If the scores are the same, the current participant belongs to the same rank group
                # as the previous participant(s). Assign the same rank value.
                ranks[original_index] = current_rank_value
            else:
                # If the score is different (it must be lower due to the descending sort),
                # this participant starts a new rank group.
                # The rank for this new group is determined by the 0-based position 'i' in the sorted list.
                # Specifically, the rank is `i + 1`. This is because 'i' participants have scores
                # greater than or equal to the current participant's score and have already been processed.
                # The rank reflects the number of people ranked higher or equal.
                current_rank_value = i + 1
                ranks[original_index] = current_rank_value

    # Print the calculated rank for each participant.
    # The `ranks` list is indexed by the original 0-based index.
    # We iterate from 0 to N-1 and print the rank corresponding to the participant
    # originally at that index (i.e., print rank of person 1, then person 2, etc.).
    for i in range(n):
        print(ranks[i])

# Standard boilerplate to ensure the main function is called when the script is executed.
if __name__ == '__main__':
    main()