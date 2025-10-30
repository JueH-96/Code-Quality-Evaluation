import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Array to store the maximum length of a run for each character 'a' through 'z'
    max_run_lengths = [0] * 26 

    # Constraints state N >= 1.
    # The loop structure below correctly handles N=1.
    
    i = 0
    while i < N:
        current_char = S[i]
        # Determine the index for this character (0 for 'a', 1 for 'b', etc.)
        char_idx = ord(current_char) - ord('a') 
        
        j = i
        # Find the end of the current run of identical characters.
        # The run consists of S[i], S[i+1], ..., S[j-1].
        while j < N and S[j] == current_char:
            j += 1
        
        # Length of the current run
        current_run_length = j - i
        
        # If this run is longer than the previously recorded maximum for this character, update it.
        if current_run_length > max_run_lengths[char_idx]:
            max_run_lengths[char_idx] = current_run_length
            
        # Move to the start of the next segment (which will be S[j], if j < N)
        i = j
        
    # The total number of distinct substrings that are repetitions of one character
    # is the sum of the maximum run lengths found for each character.
    # For example, if max_run_lengths['x'] is k, it means "x", "xx", ..., up to k 'x's
    # are the distinct substrings of this form for character 'x'. There are k such strings.
    total_count = sum(max_run_lengths)
    
    sys.stdout.write(str(total_count) + "
")

if __name__ == '__main__':
    main()