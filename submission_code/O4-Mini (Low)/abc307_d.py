def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    S = input().rstrip('
')
    
    # matched[i] = True if S[i] is a parenthesis that is part of a matched pair
    matched = [False] * N
    stack = []
    
    # First pass: find matching parentheses
    for i, ch in enumerate(S):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                matched[i] = True
                matched[j] = True
            # else unmatched ')', leave matched[i] = False
    
    # Now build the result, skipping matched parentheses and their contents
    res_chars = []
    depth = 0
    for i, ch in enumerate(S):
        if ch == '(':
            if matched[i]:
                # entering a matched parentheses scope
                depth += 1
                # skip the '(' itself
            else:
                # unmatched '(' is just a normal char
                res_chars.append(ch)
        elif ch == ')':
            if matched[i]:
                # leaving a matched parentheses scope
                depth -= 1
                # skip the ')' itself
            else:
                # unmatched ')' is just a normal char
                res_chars.append(ch)
        else:
            # a lowercase letter
            if depth == 0:
                # only include letters not inside matched parentheses
                res_chars.append(ch)
    
    # Output the reconstructed string
    print(''.join(res_chars))


if __name__ == "__main__":
    main()