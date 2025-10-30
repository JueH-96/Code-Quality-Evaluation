def main():
    import sys
    data = sys.stdin.read()
    if not data:
        return
    s = data.strip()
    # Reverse s.
    rs = s[::-1]
    # Build a combined string for KMP:
    # We want to find the longest prefix of rs that is also a suffix of s.
    # For that, we create the string: rs + '$' + s.
    combined = rs + '$' + s
    
    # Compute the prefix function (pi) for the combined string.
    n = len(combined)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and combined[i] != combined[j]:
            j = pi[j - 1]
        if combined[i] == combined[j]:
            j += 1
        pi[i] = j

    # The value at the last position indicates the length L of the longest suffix of s
    # that is a palindrome (because it equals the prefix of rs).
    L = pi[-1]
    
    # To form the shortest palindrome that starts with s, we append the reverse of the 
    # prefix of s that's not part of this suffix palindrome.
    extra = s[:len(s) - L]
    result = s + extra[::-1]
    
    sys.stdout.write(result)

if __name__ == '__main__':
    main()