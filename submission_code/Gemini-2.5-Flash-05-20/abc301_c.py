# YOUR CODE HERE
import collections

def solve():
    S = input()
    T = input()

    # Define the set of characters that '@' can be replaced with
    atcoder_chars = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

    # Get character counts for both strings
    s_counts = collections.Counter(S)
    t_counts = collections.Counter(T)

    # Iterate through all lowercase English alphabet characters
    for char_code in range(ord('a'), ord('z') + 1):
        char = chr(char_code)

        # Get counts for the current character in S and T
        # Counter returns 0 for non-existent keys, so s_counts[char] is safe
        s_char_count = s_counts[char] 
        t_char_count = t_counts[char]

        # Check if the character is one of the 'atcoder' characters
        if char in atcoder_chars:
            # If S has more of this character than T
            if s_char_count > t_char_count:
                diff = s_char_count - t_char_count
                # T must use its '@'s to provide the missing 'char's
                t_counts['@'] -= diff
                # If T doesn't have enough '@'s, it's impossible to balance
                if t_counts['@'] < 0:
                    print("No")
                    return
            # If T has more of this character than S
            elif t_char_count > s_char_count:
                diff = t_char_count - s_char_count
                # S must use its '@'s to provide the missing 'char's
                s_counts['@'] -= diff
                # If S doesn't have enough '@'s, it's impossible to balance
                if s_counts['@'] < 0:
                    print("No")
                    return
            # If counts are equal, no '@'s are needed for this character from either side.
        else:
            # If char is not an atcoder char, it cannot be substituted by '@'.
            # Therefore, its counts in S and T must be exactly equal.
            if s_char_count != t_char_count:
                print("No")
                return

    # After iterating through all regular characters, the remaining '@' counts must be equal.
    # Any remaining '@' can only be converted into other atcoder chars (which have already been balanced)
    # or remain as '@'. If the total number of available '@'s is not equal, they cannot match.
    if s_counts['@'] == t_counts['@']:
        print("Yes")
    else:
        print("No")

solve()