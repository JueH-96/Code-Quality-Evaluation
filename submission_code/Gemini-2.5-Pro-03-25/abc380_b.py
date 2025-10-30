# YOUR CODE HERE
import sys

def solve():
    # Read the input string S from standard input
    s = sys.stdin.readline().strip()
    
    # Split the string S by the delimiter '|'
    # The generation process guarantees S starts and ends with '|'.
    # Splitting by '|' results in a list where the first and last elements are empty strings,
    # and the elements in between are the sequences of dashes '-'.
    # Example: If S = "|---|-|----|", splitting gives parts = ["", "---", "-", "----", ""].
    parts = s.split('|')
    
    # We are interested in the segments containing dashes, which are located
    # between the first and last empty strings in the `parts` list.
    # We can select these segments using list slicing: parts[1:-1].
    # This effectively removes the leading and trailing empty strings.
    # For parts = ["", "---", "-", "----", ""], dash_segments will be ["---", "-", "----"].
    dash_segments = parts[1:-1] 
    
    # The problem asks to reconstruct the sequence A = (A_1, A_2, ..., A_N).
    # Each A_i corresponds to the number of dashes in the i-th segment.
    # We can find these numbers by calculating the length of each string in `dash_segments`.
    # A list comprehension is a concise way to do this.
    # For dash_segments = ["---", "-", "----"], result_a will be [len("---"), len("-"), len("----")] = [3, 1, 4].
    result_a = [len(segment) for segment in dash_segments]
    
    # Print the elements of the reconstructed sequence A.
    # The output format requires the numbers to be separated by spaces on a single line.
    # The `*` operator unpacks the elements of the list `result_a` as individual arguments to the `print` function.
    # The `print` function, by default, separates its arguments with a space.
    # For result_a = [3, 1, 4], print(*(result_a)) is equivalent to print(3, 1, 4), which outputs "3 1 4".
    print(*(result_a))

# Call the solve function to execute the logic
solve()