def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    S = data[1].strip()
    
    # Precompute, for each index, the number of consecutive '1's ending at that index.
    # This will help us quickly know, for any given index i (where S[i] == '/'),
    # how many '1's are immediately to its left.
    countLeft = [0] * n
    if S[0] == '1':
        countLeft[0] = 1
    for i in range(1, n):
        if S[i] == '1':
            countLeft[i] = countLeft[i - 1] + 1
        else:
            countLeft[i] = 0
    
    # Precompute, for each index, the number of consecutive '2's starting at that index.
    # This will let us know, for a slash in the string, how many '2's are immediately to its right.
    countRight = [0] * n
    if S[n - 1] == '2':
        countRight[n - 1] = 1
    for i in range(n - 2, -1, -1):
        if S[i] == '2':
            countRight[i] = countRight[i + 1] + 1
        else:
            countRight[i] = 0
    
    # The structure of a valid 11/22 string is:
    # "1"*m + "/" + "2"*m   (with m >= 0)
    # Thus the length of a candidate substring is 2*m + 1.
    # For each slash in the string, we can consider it as the center of a potential valid substring.
    # The maximum m we can use for that slash is determined by the number of consecutive '1's
    # immediately to its left and the number of consecutive '2's immediately to its right.
    best = 0
    for i in range(n):
        if S[i] == '/':
            left_possible = countLeft[i - 1] if i - 1 >= 0 else 0
            right_possible = countRight[i + 1] if i + 1 < n else 0
            m = min(left_possible, right_possible)  # maximum m for which both sides match
            candidate_length = 2 * m + 1
            if candidate_length > best:
                best = candidate_length
    
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()