def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1]
    
    ans = 0
    n = N
    # For every occurrence of '/', we expand outwards checking
    # how many '1's can we take from the left and how many '2's on the right.
    # The valid substring length based at a particular '/' is 1 + 2*k,
    # where k = min(count of consecutive '1's to left, count of consecutive '2's to right).
    for i, ch in enumerate(S):
        if ch != '/':
            continue
        
        # Count consecutive '1's to the left:
        count_left = 0
        j = i - 1
        while j >= 0 and S[j] == '1':
            count_left += 1
            j -= 1

        # Count consecutive '2's to the right:
        count_right = 0
        j = i + 1
        while j < n and S[j] == '2':
            count_right += 1
            j += 1

        # The valid substring can only extend as far as available in both directions.
        k = min(count_left, count_right)
        current_length = 1 + 2 * k  # 1 for '/' and k ones and k twos
        
        if current_length > ans:
            ans = current_length
            
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()