import sys

def solve():
    N_str = sys.stdin.readline() # N is not strictly needed for the logic below
    S = sys.stdin.readline().strip()

    ans_chars = []
    for char_s in S:
        if char_s == ')':
            # This is the current character from input string S.
            # We need to check if ans_chars ends with "(L...L)"
            # where L...L are letters.
            
            idx_of_opening_paren = -1 
            
            # Scan backwards through ans_chars to find a matching '('
            # and verify content.
            for k_scan_back in range(len(ans_chars) - 1, -1, -1):
                char_in_ans = ans_chars[k_scan_back]
                
                if char_in_ans == '(':
                    # Found an opening parenthesis. This is a potential match.
                    idx_of_opening_paren = k_scan_back
                    break 
                elif char_in_ans == ')':
                    # Found a ')' within the potential content.
                    # This means the content is not just letters.
                    # e.g., ans_chars is `... X ( Y ) Z` and char_s is `)`.
                    # The content for X would be `Y ) Z`. Not valid for this rule.
                    # So, current char_s (')') cannot form a simple deletable pair.
                    idx_of_opening_paren = -1 # Signal that no valid pair is formed.
                    break
                # else: char_in_ans is a letter. This is valid content. Continue scan.
            
            if idx_of_opening_paren != -1:
                # A valid '(' was found at idx_of_opening_paren,
                # and all characters between it and the original end of ans_chars were letters.
                # So, ans_chars currently looks like: ... prefix ( L1 L2 ... Lm
                # And char_s is ')'. This forms: ... prefix ( L1 L2 ... Lm )
                # This segment should be deleted. We remove '( L1 L2 ... Lm' from ans_chars.
                del ans_chars[idx_of_opening_paren:]
            else:
                # No matching '(' was found such that content was only letters,
                # or ans_chars was empty.
                # Append the current ')' (char_s) to ans_chars.
                ans_chars.append(char_s)
                
        else: # char_s is '(' or a letter
            ans_chars.append(char_s)

    sys.stdout.write("".join(ans_chars) + "
")

if __name__ == '__main__':
    solve()