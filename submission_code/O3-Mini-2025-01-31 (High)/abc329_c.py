def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Read input values.
    n = int(data[0])
    s = data[1]

    # We'll use a dictionary to store, for each character that appears in s,
    # the maximum length of a contiguous run of that character.
    max_run = {}

    # Initialize with the first character.
    current_char = s[0]
    current_length = 1

    # Process the rest of the string.
    for i in range(1, len(s)):
        if s[i] == current_char:
            current_length += 1
        else:
            # Update the maximum run for current_char.
            if current_char in max_run:
                max_run[current_char] = max(max_run[current_char], current_length)
            else:
                max_run[current_char] = current_length
            # Reset for the new character.
            current_char = s[i]
            current_length = 1

    # Update for the last group.
    if current_char in max_run:
        max_run[current_char] = max(max_run[current_char], current_length)
    else:
        max_run[current_char] = current_length

    # Each distinct repeated substring of a character c is of the form c, cc, ..., c*L
    # where L is the maximum run length for that character in the string.
    # So the number of distinct repeated substrings contributed by c is L.
    #
    # The final answer is the sum of these values for all characters.
    answer = sum(max_run.values())

    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()