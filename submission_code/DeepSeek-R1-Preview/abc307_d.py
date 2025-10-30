def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    s = list(input[1])
    
    while True:
        m = len(s)
        if m == 0:
            break
        
        # Precompute prefix sums for '(' and ')'
        open_counts = [0] * (m + 1)
        close_counts = [0] * (m + 1)
        
        for i in range(m):
            open_counts[i+1] = open_counts[i] + (1 if s[i] == '(' else 0)
            close_counts[i+1] = close_counts[i] + (1 if s[i] == ')' else 0)
        
        stack = []
        marked = [False] * m
        
        for i in range(m):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                if stack:
                    j = stack.pop()
                    # Check if between j and i, there are no other '(', ')'
                    if j + 1 > i - 1:
                        # They are adjacent or j == i, which can't happen
                        # But since j < i, j+1 <= i
                        # So if j+1 > i-1, it's j+1 == i, i-1 = j
                        # So the substring is j to i, which is exactly '()', no chars in between
                        for k in range(j, i+1):
                            marked[k] = True
                    else:
                        # Calculate the number of '(' and ')' between j+1 and i-1
                        num_open = open_counts[i] - open_counts[j+1]  # up to i-1 is i, since open_counts is 1-based
                        num_close = close_counts[i] - close_counts[j+1]
                        if num_open == 0 and num_close == 0:
                            for k in range(j, i+1):
                                marked[k] = True
        
        # Build the new string
        new_s = []
        for i in range(m):
            if not marked[i]:
                new_s.append(s[i])
        
        if len(new_s) == m:
            break
        s = new_s
    
    print(''.join(s))

if __name__ == "__main__":
    main()