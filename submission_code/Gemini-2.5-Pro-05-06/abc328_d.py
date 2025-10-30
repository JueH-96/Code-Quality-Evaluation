import sys

def main():
    S = sys.stdin.readline().strip()
    
    # res_chars will act as a stack to build the result string
    res_chars = []
    
    for char_s in S:
        res_chars.append(char_s)
        
        # After appending a character, check if the end of res_chars forms "ABC".
        # This needs to be a loop because removing an "ABC" might reveal
        # another "ABC" at the end. e.g., "XYABC" becomes "XY", if "XY" was "ZABC",
        # that needs to be handled too. If res_chars was "ABCAB" and 'C' is added,
        # it becomes "ABCABC". First "ABC" removal leaves "ABC". Second leaves [].
        while len(res_chars) >= 3:
            # Check if the last three characters are 'A', 'B', 'C'
            if res_chars[-3] == 'A' and \
               res_chars[-2] == 'B' and \
               res_chars[-1] == 'C':
                # If they form "ABC", pop them
                res_chars.pop()  # Pop 'C'
                res_chars.pop()  # Pop 'B'
                res_chars.pop()  # Pop 'A'
            else:
                # If the last three characters do not form "ABC",
                # then no more reduction is possible with the current suffix.
                # Break the while loop and process the next character from S.
                break
                
    # Join the characters in res_chars to form the final string and print it.
    sys.stdout.write("".join(res_chars) + "
")

if __name__ == '__main__':
    main()