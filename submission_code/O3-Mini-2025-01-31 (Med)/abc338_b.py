def main():
    import sys
    input_str = sys.stdin.read().strip()
    if not input_str:
        return
    s = input_str.strip()
    
    frequency = {}
    for ch in s:
        frequency[ch] = frequency.get(ch, 0) + 1
    
    max_freq = max(frequency.values())
    # find all characters with maximum frequency and then select the earliest one in alphabetical order
    result = min([ch for ch in frequency if frequency[ch] == max_freq])
    sys.stdout.write(result)

if __name__ == "__main__":
    main()