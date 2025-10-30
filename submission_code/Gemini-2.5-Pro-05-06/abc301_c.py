import collections
import sys

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    s_counts = collections.Counter(S)
    t_counts = collections.Counter(T)

    s_at_val = s_counts.pop('@', 0)
    t_at_val = t_counts.pop('@', 0)

    atcoder_chars = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

    possible = True
    # Iterate over all lowercase English letters
    for i in range(ord('a'), ord('z') + 1):
        char_c = chr(i)
        
        sc = s_counts.get(char_c, 0)
        tc = t_counts.get(char_c, 0)

        if sc == tc:
            continue
        
        diff = abs(sc - tc) # Number of characters needed to make counts equal

        if sc < tc:
            # S needs 'diff' copies of char_c
            if char_c not in atcoder_chars:
                possible = False # Cannot form char_c
                break
            s_at_val -= diff # Use s_at_val cards from S
            if s_at_val < 0:
                possible = False # Not enough '@' cards in S
                break
        else: # sc > tc
            # T needs 'diff' copies of char_c
            if char_c not in atcoder_chars:
                possible = False # Cannot form char_c
                break
            t_at_val -= diff # Use t_at_val cards from T
            if t_at_val < 0:
                possible = False # Not enough '@' cards in T
                break
    
    if not possible:
        sys.stdout.write("No
")
        return

    # If loop completed and 'possible' is True, then:
    # 1. All character count differences (for non-'@' chars) could be resolved.
    # 2. s_at_val and t_at_val are current counts of remaining '@' cards, both >= 0.
    # These remaining '@' cards must also be converted.
    # For the final multisets to be identical, we need s_at_val == t_at_val.
    # If they are equal (say, K), both S and T can convert their K '@' cards
    # into K copies of 'a' (or any other char in atcoder_chars), maintaining equality.
    if s_at_val == t_at_val:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

if __name__ == '__main__':
    main()