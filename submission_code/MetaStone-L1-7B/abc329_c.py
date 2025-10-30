def count_repeated_substrings():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1] if len(data) > 1 else ''
    
    if N == 0:
        print(0)
        return
    
    from collections import defaultdict
    max_run = defaultdict(int)
    total = 0
    current_char = S[0]
    current_run_length = 1
    
    for i in range(1, N):
        if S[i] == current_char:
            current_run_length += 1
        else:
            # Process the current run
            max_run[current_char] = max(max_run[current_char], current_run_length)
            total += (current_run_length - max_run[current_char])
            # Reset for next character
            current_char = S[i]
            current_run_length = 1
    
    # Process the last run
    max_run[current_char] = max(max_run[current_char], current_run_length)
    total += (current_run_length - max_run[current_char])
    
    print(total)

count_repeated_substrings()