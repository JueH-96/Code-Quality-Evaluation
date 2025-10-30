def replace_wa_with_ac(s):
    while 'WA' in s:
        # Find the leftmost occurrence of 'WA'
        index = s.index('WA')
        # Replace it with 'AC'
        s = s[:index] + 'AC' + s[index + 2:]
    return s

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    result = replace_wa_with_ac(S)
    print(result)