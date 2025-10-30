def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    t = int(input_data[0])
    idx = 1
    answers = []
    
    for _ in range(t):
        n = int(input_data[idx]); idx+=1
        k = int(input_data[idx]); idx+=1
        s = input_data[idx]; idx+=1
        
        black_positions = [i for i, ch in enumerate(s) if ch == 'B']
        
        if not black_positions:
            # No black cells at all
            answers.append(0)
            continue
        
        # Greedy covering from left to right
        count = 1
        current_cover_end = black_positions[0] + k - 1
        
        for pos in black_positions:
            if pos > current_cover_end:
                count += 1
                current_cover_end = pos + k - 1
        
        answers.append(count)
    
    # Output results
    print('
'.join(map(str, answers)))

# Do not forget to call main
main()