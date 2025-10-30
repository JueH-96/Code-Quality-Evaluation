import sys
import itertools

def solve():
    # Read the five integer scores for problems A, B, C, D, E
    # input().split() reads the line and splits it by whitespace into a list of strings.
    # map(int, ...) converts these strings to integers.
    # The values are then assigned to a, b, c, d, e.
    a, b, c, d, e = map(int, sys.stdin.readline().split())

    # Define the problem letters in alphabetical order
    problem_letters = ['A', 'B', 'C', 'D', 'E']
    
    # Create a dictionary to map each problem letter to its corresponding score
    problem_scores_map = {
        'A': a,
        'B': b,
        'C': c,
        'D': d,
        'E': e
    }

    # This list will store tuples of (score, participant_name).
    # Storing score first is useful for sorting.
    participants_data = []

    # Generate all non-empty subsequences of problem_letters.
    # A non-empty subsequence corresponds to a participant.
    # The length of a subsequence can range from 1 (e.g., 'A') to 5 (e.g., 'ABCDE').
    # itertools.combinations(iterable, r) generates combinations of length r from the iterable.
    # The combinations are yielded in lexicographical sort order of the input iterable.
    # This means the participant names generated will have their letters in alphabetical order (e.g., "AB" not "BA").
    for r in range(1, len(problem_letters) + 1):
        for combo_tuple in itertools.combinations(problem_letters, r):
            # Form the participant's name by joining the letters in the combination.
            # Example: ('A', 'B', 'C') becomes "ABC".
            name = "".join(combo_tuple)
            
            # Calculate the total score for this participant.
            # Iterate through each problem letter in their name and sum its score.
            current_score = 0
            for char_in_name in name:
                current_score += problem_scores_map[char_in_name]
            
            # Store the calculated score and the participant's name.
            # We store it as (current_score, name) so that when we sort,
            # Python's default tuple sorting mechanism can be leveraged.
            participants_data.append((current_score, name))

    # Sort the participants based on the problem's criteria:
    # 1. Primary sort key: Score in descending order (largest to smallest).
    # 2. Secondary sort key: Participant name in lexicographical (alphabetical) ascending order for ties.
    
    # Python's list.sort() method with a `key` function is ideal.
    # For a tuple `(score, name)`, `lambda x: (-x[0], x[1])` creates a new tuple for sorting:
    #   - `-x[0]` (negative of the score) ensures that higher scores become smaller negative numbers,
    #     so they come first when sorted in ascending order. This effectively sorts scores in descending order.
    #   - `x[1]` (the name) is used as the secondary sort key. Since it's a string,
    #     it will be sorted lexicographically in ascending order, which is exactly what's required for ties.
    participants_data.sort(key=lambda x: (-x[0], x[1]))

    # Print the names of the participants in the sorted order.
    # Each name should be on a new line.
    for score, name in participants_data:
        print(name)

# Call the solve function to execute the program.
if __name__ == '__main__':
    solve()