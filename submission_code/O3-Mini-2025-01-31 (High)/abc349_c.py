def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].strip()  # lowercase letters
    T = data[1].strip()  # uppercase letters, length 3

    # Helper function to check whether pattern (list of characters) is a subsequence of S.
    def is_subsequence(s, pattern):
        i = 0
        for ch in s:
            if i < len(pattern) and ch == pattern[i]:
                i += 1
            if i == len(pattern):
                return True
        return False

    # Method 1: Check if T can be derived by taking a subsequence of length 3 from S.
    pattern1 = [char.lower() for char in T]
    method1 = is_subsequence(S, pattern1)

    # Method 2: Check if T can be derived by taking a subsequence of length 2 from S and appending 'X'.
    method2 = False
    if T[2] == 'X':  # Only eligible if the last character is 'X'
        pattern2 = [T[0].lower(), T[1].lower()]
        method2 = is_subsequence(S, pattern2)

    # Print result based on if any of the two methods is possible.
    sys.stdout.write("Yes" if method1 or method2 else "No")

if __name__ == '__main__':
    main()