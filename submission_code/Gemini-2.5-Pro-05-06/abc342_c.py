import sys

def main():
    N = int(sys.stdin.readline())
    S_original_str = sys.stdin.readline().strip()
    Q = int(sys.stdin.readline())

    # ascii_val_of_a is used to convert characters to 0-25 indices
    ascii_val_of_a = ord('a')

    # current_transformation_map[i] stores the character that chr(ascii_val_of_a + i)
    # (the i-th letter of the alphabet, 0-indexed) currently maps to.
    # Initially, each character maps to itself.
    current_transformation_map = [chr(ascii_val_of_a + i) for i in range(26)]

    for _ in range(Q):
        # Read the operation characters c_char and d_char
        line_parts = sys.stdin.readline().split()
        c_char = line_parts[0]
        d_char = line_parts[1]
        
        # Update the transformation map:
        # For every original character (represented by index i from 0 to 25):
        #   If its current transformed value is c_char,
        #   then its new transformed value becomes d_char.
        for i in range(26): 
            if current_transformation_map[i] == c_char:
                current_transformation_map[i] = d_char
    
    # After all operations, current_transformation_map[i] holds the final character
    # that the original character chr(ascii_val_of_a + i) has become.

    # Construct the result string by applying the final transformation map
    # to each character of the original string S_original_str.
    # Using a list of characters and then "".join() is efficient in Python.
    result_char_list = []
    for char_in_S in S_original_str:
        original_char_index = ord(char_in_S) - ascii_val_of_a
        mapped_char = current_transformation_map[original_char_index]
        result_char_list.append(mapped_char)
    
    sys.stdout.write("".join(result_char_list) + "
")

if __name__ == '__main__':
    main()