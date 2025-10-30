import functools

@functools.lru_cache(maxsize=None)
def generate_carpet_rows_for_level(k: int): # k is the level of the carpet
    # Base case: A level-0 carpet is a single black cell.
    if k == 0:
        return ["#"]

    # Recursively get the pattern for a level-(k-1) carpet.
    # This will be a list of strings, where each string is a row.
    prev_level_carpet_pattern = generate_carpet_rows_for_level(k - 1)
    
    # Determine the side length of the level-(k-1) carpet.
    # Since it's a square, number of rows = number of columns.
    side_length_of_prev_level_carpet = len(prev_level_carpet_pattern)
    
    # The central block of a level-K carpet is white. This white block has
    # the same dimensions as a level-(k-1) carpet.
    # We need one row of this white block pattern to use in concatenation.
    white_block_row_segment = "." * side_length_of_prev_level_carpet
    
    # This list will store the rows of the new level-K carpet.
    current_level_carpet_rows = []
    
    # A level-K carpet is made of 3x3 sub-blocks.
    # Each sub-block is either a level-(k-1) carpet or a white block.

    # Construct the top third of the rows for the level-K carpet:
    # This part consists of three level-(k-1) carpets horizontally.
    # Structure: prev_pattern | prev_pattern | prev_pattern
    for i in range(side_length_of_prev_level_carpet):
        # prev_level_carpet_pattern[i] is the i-th row of the level-(k-1) carpet.
        row_segment_from_prev_pattern = prev_level_carpet_pattern[i]
        new_row = row_segment_from_prev_pattern + row_segment_from_prev_pattern + row_segment_from_prev_pattern
        current_level_carpet_rows.append(new_row)
        
    # Construct the middle third of the rows:
    # This part consists of a level-(k-1) carpet, then a white block, then another level-(k-1) carpet.
    # Structure: prev_pattern | white_block | prev_pattern
    for i in range(side_length_of_prev_level_carpet):
        row_segment_from_prev_pattern = prev_level_carpet_pattern[i]
        # white_block_row_segment is used here for the central part.
        new_row = row_segment_from_prev_pattern + white_block_row_segment + row_segment_from_prev_pattern
        current_level_carpet_rows.append(new_row)

    # Construct the bottom third of the rows:
    # This part has the same structure as the top third.
    # Structure: prev_pattern | prev_pattern | prev_pattern
    for i in range(side_length_of_prev_level_carpet):
        row_segment_from_prev_pattern = prev_level_carpet_pattern[i]
        new_row = row_segment_from_prev_pattern + row_segment_from_prev_pattern + row_segment_from_prev_pattern
        current_level_carpet_rows.append(new_row)
        
    return current_level_carpet_rows

# Read the input integer N.
N = int(input())

# Generate the carpet for level N.
# The result is a list of strings, each representing a row of the carpet.
final_carpet_rows = generate_carpet_rows_for_level(N)

# Print each row of the generated carpet.
for carpet_row_string in final_carpet_rows:
    print(carpet_row_string)