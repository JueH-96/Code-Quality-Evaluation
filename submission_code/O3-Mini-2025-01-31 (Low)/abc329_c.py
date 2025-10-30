def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    s = data[1]

    # track maximum consecutive count for each character
    max_counts = {chr(c): 0 for c in range(ord('a'), ord('z')+1)}

    # Process the string to compute consecutive groups
    if n > 0:
        current_char = s[0]
        current_count = 1
        for ch in s[1:]:
            if ch == current_char:
                current_count += 1
            else:
                max_counts[current_char] = max(max_counts[current_char], current_count)
                current_char = ch
                current_count = 1
        # update for the last group
        max_counts[current_char] = max(max_counts[current_char], current_count)

    # Sum up maximum counts (each letter's maximum gives unique substrings: letter*k for k in 1..max_count)
    result = sum(max_counts[ch] for ch in max_counts)
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()