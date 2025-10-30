def main():
    S = input().strip()
    # cur_p is the 1-based position in the infinite pattern ioioio...
    cur_p = 1
    for c in S:
        # We want to match S character c at the earliest position j >= cur_p
        # in the pattern where j%2==1=>'i', j%2==0=>'o'.
        if c == 'i':
            # need j odd
            if cur_p % 2 == 1:
                j = cur_p
            else:
                j = cur_p + 1
        else:  # c == 'o'
            # need j even
            if cur_p % 2 == 0:
                j = cur_p
            else:
                j = cur_p + 1
        # next search will start after j
        cur_p = j + 1

    # cur_p-1 is the last used index in the pattern
    used = cur_p - 1
    # the final string length must be even
    if used % 2 == 1:
        used += 1

    # number of insertions = final length - original length
    print(used - len(S))

# call main
main()